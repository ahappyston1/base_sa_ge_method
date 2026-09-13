"""Independent REV14 CIFAR10 experiment; REV13 training remains untouched.

Teacher classification follows local EMA. Geometry uses a frozen SAME-ROUND global
teacher for both support and queries. Raw images and labels stay on each client;
only compressed labeled-support banks are merged. Test labels are evaluation only.
"""
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from Dataset.dataset import classify_label, partition_train, Indices2Dataset_labeled, Indices2Dataset_unlabeled_fixmatch
from Dataset.sample_dirichlet import clients_indices
from Dataset.normalize import to_tensor_normalize
from Model.resnet import ResNet
from trusted_multi import Learner, schedule, ema_update, no_bn_tracking, fit_bank, merge_banks, make_targets, objective, aggregate

REVISION='trusted_multi_v1'


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def make_model(device):
    backbone=ResNet(resnet_size=8,scaling=4,save_activations=False,group_norm_num_groups=None,
                    freeze_bn=False,freeze_bn_affine=False,num_classes=10).to(device)
    backbone.eval()
    with torch.no_grad():dim=backbone(torch.zeros(1,3,32,32,device=device))[0].shape[1]
    return Learner(backbone,dim).to(device)


def partition(dataset,args):
    labeled,unlabeled=partition_train(classify_label(dataset,10),500,rng=np.random.RandomState(args.partition_seed))
    def divide(rows):
        parts=clients_indices(rows,10,args.num_clients,args.alpha,seed=args.partition_seed)
        return [[int(i) for i in ids] for ids in parts]
    labeled,unlabeled=divide(labeled),divide(unlabeled)
    unlabeled=[u+l for u,l in zip(unlabeled,labeled)]
    return labeled,unlabeled


@torch.no_grad()
def collect_bank(anchor,dataset,indices,args,device):
    transform=to_tensor_normalize('CIFAR10');features=[];labels=[]
    # Stable ID order makes support/query disjoint and deterministic within class.
    ids=sorted(set(indices))
    for start in range(0,len(ids),128):
        rows=[dataset[i] for i in ids[start:start+128]]
        x=torch.stack([transform(image) for image,_ in rows]).to(device)
        features.append(anchor(x)[0]);labels.extend(int(y) for _,y in rows)
    if not features:raise ValueError('A participating client has no labeled reference')
    return fit_bank(torch.cat(features),torch.tensor(labels,device=device),10,args.tm_prototypes,args.tm_min_support)


def endless(loader):
    while True:
        yield from loader


def train_client(student,teacher,anchor,bank,global_bank,labeled,unlabeled,args,r,device,prior=None):
    student.train();teacher.eval();lr,ramp=schedule(r,args)
    opt=torch.optim.SGD(student.parameters(),lr=lr,momentum=.9,weight_decay=1e-4)
    kw=dict(num_workers=args.num_workers,pin_memory=device.type=='cuda')
    if args.num_workers:kw.update(persistent_workers=True,prefetch_factor=4)
    batch=args.batch_size_local_labeled_fixmatch
    if len(unlabeled)<batch*args.mu:raise ValueError('Unlabeled client too small for a full batch')
    ll=DataLoader(labeled,batch_size=batch,shuffle=True,drop_last=True,**kw)
    ul=DataLoader(unlabeled,batch_size=batch*args.mu,shuffle=True,drop_last=True,**kw)
    li,ui=endless(ll),endless(ul)
    # Exact REV13 convention: steps are based on labeled batch size, not mu*batch.
    steps=args.local_epochs*max(1,len(unlabeled)//batch)
    stats={k:0. for k in ['L_sup','L_A','L_B','L_proto','L_feature','loss','a_count','b_count','c_count','a_correct','b_correct','c_correct','a_wrong_weight','u_visits','geometry_mix','geometry_valid','target_entropy','teacher_correct','fused_correct','correction_good','correction_bad','feature_std']}
    if prior is None:prior=torch.full((10,),.1,device=device)
    counts=torch.bincount(torch.tensor([y for _,y in labeled.client_dataset[:labeled.client_dataset_original_len]],device=device),minlength=10).float()
    local_prior=(counts+1)/(counts.sum()+10)
    for _ in range(steps):
        x,y=next(li);uw,us,diagnostic_gt,_ids=next(ui)
        x,y,uw,us=x.to(device),y.to(device),uw.to(device),us.to(device)
        labeled_output=student(x)
        with torch.no_grad():
            # Only labeled/weak student views update the inference BN buffers.
            student(uw)
            _,teacher_logits,teacher_projection=teacher(uw)
            probs=teacher_logits.softmax(-1)
            geometry_z=anchor(uw)[0]
            prior=.99*prior+.01*probs.mean(0)
            target=make_targets(probs,geometry_z,bank,global_bank,ramp,args,(local_prior,prior))
        with no_bn_tracking(student):strong_output=student(us)
        loss,terms=objective(student,labeled_output,strong_output,teacher_projection,y,target,global_bank,ramp,args)
        if not torch.isfinite(loss):raise FloatingPointError(f'Non-finite loss at round {r}')
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(),float('inf'),error_if_nonfinite=True)
        opt.step();ema_update(teacher,student,args.tm_local_ema)
        with torch.no_grad():
            # Hidden GT is consumed ONLY here, after optimizer.step, for read-only logs.
            correct=target['pred']==diagnostic_gt.to(device)
            for key,value in terms.items():stats[key]+=float(value)
            stats['loss']+=float(loss)
            for bucket in ['a','b','c']:stats[bucket+'_count']+=int(target[bucket].sum())
            for bucket in ['a','b','c']:stats[bucket+'_correct']+=int((target[bucket]&correct).sum())
            stats['a_wrong_weight']+=float((target['weight']*target['a']*~correct).sum())*ramp
            stats['u_visits']+=len(uw)
            stats['geometry_mix']+=float(target['gamma'].sum())
            stats['geometry_valid']+=int((target['gamma']>0).sum())
            stats['target_entropy']+=float(-(target['q']*target['q'].clamp_min(1e-8).log()).sum())
            teacher_correct=probs.argmax(-1)==diagnostic_gt.to(device)
            stats['teacher_correct']+=int(teacher_correct.sum());stats['fused_correct']+=int(correct.sum())
            stats['correction_good']+=int((~teacher_correct&correct).sum())
            stats['correction_bad']+=int((teacher_correct&~correct).sum())
            stats['feature_std']+=float(torch.nn.functional.normalize(strong_output[2],dim=-1).std(dim=0,unbiased=False).mean())
    stats['steps']=steps
    return stats,prior.detach()


@torch.no_grad()
def evaluate(model,dataset,device,batch):
    model.eval();correct=total=0
    for x,y in DataLoader(dataset,batch_size=batch,shuffle=False,num_workers=0):
        logits=model(x.to(device))[1];correct+=int((logits.argmax(-1)==y.to(device)).sum());total+=len(y)
    return correct/max(total,1)


def atomic_save(value,path):
    temporary=path.with_suffix('.tmp');torch.save(value,temporary);os.replace(temporary,path)


def append_csv(path,row):
    exists=path.exists()
    with path.open('a',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(row))
        if not exists:w.writeheader()
        w.writerow(row)


def truncate_csv(path,round_idx):
    if not path.exists():return
    with path.open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=[r for r in reader if int(r['round'])<=round_idx]
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def run(args, *, train_data=None, test_data=None, device=None):
    seed_all(args.seed)
    if device is None:
        if not torch.cuda.is_available():raise RuntimeError('Full experiment requires CUDA; run CPU unit/smoke tests separately')
        device=torch.device(f'cuda:{args.gpu_id}')
    elif train_data is None or test_data is None:
        raise ValueError('Device override is reserved for injected synthetic smoke datasets')
    total_rounds=args.tm_main_rounds+args.tm_tail_rounds
    resume=Path(args.resume).resolve() if args.resume else None
    run_id=args.run_id or f'rev14_trusted_multi_s{args.seed}_{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}'
    out=resume.parent if resume else Path('results/CIFAR10/runs')/f'{run_id}_a{args.alpha:g}'
    out.mkdir(parents=True,exist_ok=True)
    if not resume and any(out.iterdir()):raise FileExistsError(f'Refusing to overwrite existing run: {out}')
    train=train_data if train_data is not None else CIFAR10(args.path_cifar10,train=True,download=True,transform=None)
    test=test_data if test_data is not None else CIFAR10(args.path_cifar10,train=False,download=True,transform=to_tensor_normalize('CIFAR10'))
    lab_ids,u_ids=partition(train,args)
    digest=hashlib.sha256(json.dumps([lab_ids,u_ids]).encode()).hexdigest()
    student=make_model(device);teacher=copy.deepcopy(student).eval()
    for p in teacher.parameters():p.requires_grad_(False)
    rng=np.random.RandomState(args.sample_seed)
    priors={};history=None;start=1;accuracy=[]
    config=dict(vars(args));config.update(method_rev=REVISION,num_labeled_per_class=500,partition_sha256=digest,total_rounds=total_rounds)
    if resume:
        # Checkpoints must be trusted local training artifacts (contain RNG state).
        ckpt=torch.load(resume,map_location='cpu',weights_only=False)
        if ckpt['revision']!=REVISION:raise ValueError('Cannot resume a different method revision')
        ignored={'resume','run_id','gpu_id','num_workers','config'}
        changes={k for k in config if k not in ignored and config[k]!=ckpt['config'].get(k)}
        if changes:raise ValueError(f'Resume configuration changed: {sorted(changes)}')
        student.load_state_dict(ckpt['student']);teacher.load_state_dict(ckpt['teacher'])
        priors={int(k):v.to(device) for k,v in ckpt['priors'].items()};history=ckpt['history']
        accuracy=ckpt['accuracy'];start=ckpt['round']+1
        rng.set_state(ckpt['sampling_rng']);random.setstate(ckpt['python_rng']);np.random.set_state(ckpt['numpy_rng'])
        torch.set_rng_state(ckpt['torch_rng'])
        if torch.cuda.is_available():torch.cuda.set_rng_state_all(ckpt['cuda_rng'])
        for filename in ['acc.csv','metrics.csv','client_updates.csv']:truncate_csv(out/filename,start-1)
    else:
        (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
        (out/'partition.json').write_text(json.dumps(dict(labeled=lab_ids,unlabeled=u_ids)),encoding='utf-8')
    print(f'[REV14] {out}; rounds={total_rounds}; no pretrained weights; test student only',flush=True)
    for r in range(start,total_rounds+1):
        began=time.time();online=[int(k) for k in rng.choice(args.num_clients,args.num_online_clients,replace=False)]
        anchor=copy.deepcopy(teacher).eval()
        # One reference refresh for participating clients BEFORE local optimization.
        banks={k:collect_bank(anchor,train,lab_ids[k],args,device) for k in online}
        global_bank=merge_banks(list(banks.values()),args.tm_prototypes)
        states=[];weights=[];logs=[]
        for k in online:
            local=copy.deepcopy(student);local_teacher=copy.deepcopy(teacher)
            ld=Indices2Dataset_labeled(train,'CIFAR10');ld.load(lab_ids[k])
            ud=Indices2Dataset_unlabeled_fixmatch(train,'CIFAR10');ud.load(u_ids[k])
            log,priors[k]=train_client(local,local_teacher,anchor,banks[k],global_bank,ld,ud,args,r,device,priors.get(k))
            states.append({key:value.detach().clone() for key,value in local.state_dict().items()})
            weights.append(len(set(u_ids[k])|set(lab_ids[k])));logs.append(log)
            del local,local_teacher,ld,ud
        history,updates=aggregate(student,states,weights,history,args.tm_clip_factor)
        ema_update(teacher,student,args.tm_server_ema)
        # Evaluation never controls routing, losses, reference selection, LR or aggregation.
        acc=evaluate(student,test,device,args.batch_size_test);accuracy.append(acc)
        steps=sum(x['steps'] for x in logs);visits=sum(x['u_visits'] for x in logs)
        row=dict(round=r,phase='tail' if r>args.tm_main_rounds else 'main',acc=acc,lr=schedule(r,args)[0],ramp=schedule(r,args)[1])
        for name in ['loss','L_sup','L_A','L_B','L_proto','L_feature']:row[name]=sum(x[name] for x in logs)/steps
        for b in ['a','b','c']:
            count=sum(x[b+'_count'] for x in logs);row[b+'_ratio']=count/visits
            row[b+'_prec']=sum(x[b+'_correct'] for x in logs)/max(count,1)
        for key in ['a_wrong_weight','geometry_mix','geometry_valid','target_entropy','teacher_correct','fused_correct','correction_good','correction_bad']:row[key]=sum(x[key] for x in logs)/visits
        row['feature_std']=sum(x['feature_std'] for x in logs)/steps
        row['global_valid_classes']=int(global_bank['valid'].sum())
        row['global_prototypes']=int((global_bank['mass']>0).sum())
        row['prototype_upload_bytes']=sum(v.numel()*v.element_size() for b in banks.values() for v in b.values())
        row.update(updates);row.update(seconds=time.time()-began,steps=steps,u_visits=visits,best_acc=max(accuracy),best_round=accuracy.index(max(accuracy))+1)
        if abs(sum(row[k] for k in ['L_sup','L_A','L_B','L_proto','L_feature'])-row['loss'])>1e-5:raise AssertionError('Loss logging mismatch')
        append_csv(out/'metrics.csv',row);append_csv(out/'acc.csv',dict(round=r,acc=acc))
        for k,w,log in zip(online,weights,logs):append_csv(out/'client_updates.csv',dict(round=r,client=k,fedavg_weight=w/sum(weights),**log))
        checkpoint=dict(revision=REVISION,config=config,round=r,student=student.state_dict(),teacher=teacher.state_dict(),
                        priors={k:v.cpu() for k,v in priors.items()},history=history,accuracy=accuracy,
                        sampling_rng=rng.get_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
        atomic_save(checkpoint,out/'checkpoint.pt')
        if r==args.tm_main_rounds:atomic_save(checkpoint,out/'main_end.pt')
        # Fixed terminal student is primary; peak remains a descriptive diagnostic.
        with (out/'train.log').open('a',encoding='utf-8') as f:
            f.write(time.strftime('%Y-%m-%d %H:%M:%S')+' '+json.dumps(row)+'\n')
        print(f'Round {r}/{total_rounds} acc={acc:.4f} best={max(accuracy):.4f} A/B/C={row["a_ratio"]:.3f}/{row["b_ratio"]:.3f}/{row["c_ratio"]:.3f} loss={row["loss"]:.4f}',flush=True)
        del states,anchor,banks,global_bank


if __name__=='__main__':
    from options import args_parser
    run(args_parser())
