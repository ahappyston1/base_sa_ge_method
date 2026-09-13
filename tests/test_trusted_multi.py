"""CPU checks of gradients, hidden-label isolation, geometry and checkpoint state."""
import copy
import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from options import args_parser
from trusted_multi import Learner, fit_bank, merge_banks, bank_logits, make_targets, objective, schedule, aggregate, ema_update, no_bn_tracking
from trusted_multi_runner import train_client, seed_all, atomic_save, make_model, collect_bank, partition, evaluate, truncate_csv, append_csv, run
from Dataset.dataset import Indices2Dataset_labeled, Indices2Dataset_unlabeled_fixmatch

torch.set_num_threads(2)

def args():
    with patch.object(sys,'argv',['test','--config',str(ROOT/'configs/experiment_trusted_multi.yaml')]):
        a=args_parser()
    a.num_workers=0;a.local_epochs=1;a.batch_size_local_labeled_fixmatch=4
    return a


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__();self.bn=nn.BatchNorm2d(3);self.linear=nn.Linear(3,8);self.head=nn.Linear(8,10)
    def forward(self,x):
        z=self.linear(self.bn(x).mean((2,3)));return z,self.head(z)


class Pictures(Dataset):
    def __init__(self,n=32):
        self.images=[Image.fromarray(np.full((32,32,3),20+(i*7)%220,dtype=np.uint8)) for i in range(n)]
    def __len__(self):return len(self.images)
    def __getitem__(self,i):return self.images[i],i%2


class Tests(unittest.TestCase):
    def setUp(self):seed_all(7);self.a=args()

    def bank(self):
        z=torch.cat([torch.tensor([[1.,0.,0.]]).repeat(20,1),torch.tensor([[0.,1.,0.]]).repeat(20,1)])
        return fit_bank(z,torch.tensor([0]*20+[1]*20),10,3,8)

    def test_support_and_missing_class(self):
        b=self.bank();self.assertEqual(int(b['valid'].sum()),2)
        self.assertEqual(int(b['mass'].sum()),20) # query half never enters centers
        self.assertTrue((b['trust'][:2]>0).all())
        merged=merge_banks([b,b],3)
        logits=bank_logits(torch.tensor([[1.,0.,0.]]),merged)
        self.assertEqual(int(logits.argmax()),0);self.assertTrue(torch.isfinite(logits).all())

    def test_duplicate_prototypes_have_no_class_count_advantage(self):
        b=self.bank();before=bank_logits(torch.eye(3),b)
        b['centers'][0,1]=b['centers'][0,0];b['mass'][0,1]=b['mass'][0,0]
        torch.testing.assert_close(before,bank_logits(torch.eye(3),b))

    def test_unknown_geometry_is_noop_and_targets_normalized(self):
        b=self.bank();b['valid'].zero_();b['trust'].zero_()
        p=torch.softmax(torch.randn(5,10),-1)
        t=make_targets(p,torch.randn(5,3),b,b,1,self.a)
        torch.testing.assert_close(t['q'],p)
        self.assertTrue((t['a'].int()+t['b'].int()+t['c'].int()==1).all())
        self.assertFalse(t['q'].requires_grad)

    def test_bn_tracking_and_ema(self):
        s=Learner(TinyBackbone(),8).train();t=copy.deepcopy(s)
        old=s.backbone.bn.running_mean.clone();count=s.backbone.bn.num_batches_tracked.clone()
        with no_bn_tracking(s):s(torch.randn(4,3,8,8))
        torch.testing.assert_close(old,s.backbone.bn.running_mean)
        torch.testing.assert_close(count,s.backbone.bn.num_batches_tracked)
        s(torch.randn(4,3,8,8));ema_update(t,s,.9)
        self.assertEqual(t.backbone.bn.num_batches_tracked.dtype,torch.int64)
        torch.testing.assert_close(t.backbone.bn.running_mean,s.backbone.bn.running_mean)

    def test_tail_does_not_stretch_main_schedule(self):
        a=self.a;b=copy.deepcopy(a);b.tm_tail_rounds=75
        self.assertEqual([schedule(r,a) for r in range(1,301)],[schedule(r,b) for r in range(1,301)])
        self.assertEqual(schedule(375,b)[0],.0001)

    def test_clipping_and_integer_buffers(self):
        m=Learner(TinyBackbone(),8);state=copy.deepcopy(m.state_dict())
        state['backbone.linear.weight']+=100
        state['backbone.bn.num_batches_tracked'].fill_(5)
        old=m.backbone.linear.weight.clone()
        history,stats=aggregate(m,[state],[1],1,2)
        self.assertEqual(stats['clip_fraction'],1)
        self.assertLessEqual(float((m.backbone.linear.weight-old).detach().norm()),2.00001)
        self.assertEqual(int(m.backbone.bn.num_batches_tracked),5)

    def test_hidden_labels_do_not_change_training(self):
        data=Pictures();ld=Indices2Dataset_labeled(data);ld.load(list(range(16)))
        ud=Indices2Dataset_unlabeled_fixmatch(data);ud.load(list(range(32)))
        class WrongLabels(Dataset):
            def __len__(self):return len(ud)
            def __getitem__(self,i):
                w,s,y,idx=ud[i];return w,s,(y+4)%10,idx
        seed_all(11);base=Learner(TinyBackbone(),8)
        anchor=copy.deepcopy(base).eval()
        bank=collect_bank(anchor,data,list(range(16)),self.a,torch.device('cpu'))
        def train(u):
            seed_all(17);student=copy.deepcopy(base);teacher=copy.deepcopy(anchor)
            for p in teacher.parameters():p.requires_grad_(False)
            log,prior=train_client(student,teacher,anchor,bank,bank,ld,u,self.a,60,torch.device('cpu'))
            return student,log
        s1,l1=train(ud);s2,l2=train(WrongLabels())
        for k,v in s1.state_dict().items():torch.testing.assert_close(v,s2.state_dict()[k],rtol=0,atol=0)
        self.assertGreater(l1['L_feature'],0)
        self.assertNotEqual(float(s1.projector[0].weight.detach().sum()),float(base.projector[0].weight.detach().sum()))
        self.assertAlmostEqual(l1['loss'],sum(l1[k] for k in ['L_sup','L_A','L_B','L_proto','L_feature']),places=5)
        self.assertTrue(all(torch.isfinite(p).all() for p in s1.parameters()))

    def test_real_backbone_forward_backward(self):
        model=make_model(torch.device('cpu')).train()
        z,logits,p=model(torch.randn(4,3,32,32))
        (logits.square().mean()+model.predictor(p).square().mean()).backward()
        self.assertEqual(tuple(logits.shape),(4,10));self.assertTrue(torch.isfinite(model.backbone.conv1.weight.grad).all())

    def test_atomic_checkpoint_and_log_rewind(self):
        model=Learner(TinyBackbone(),8)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checkpoint.pt';atomic_save(dict(student=model.state_dict(),torch_rng=torch.get_rng_state()),path)
            state=torch.load(path,weights_only=False)
            torch.testing.assert_close(state['student']['projector.0.weight'],model.projector[0].weight)
            log=Path(tmp)/'acc.csv'
            for r in range(1,4):append_csv(log,dict(round=r,acc=.5))
            truncate_csv(log,2)
            self.assertEqual(len(log.read_text().splitlines()),3)

    def test_all_loss_branches_backward_and_teacher_detached(self):
        model=Learner(TinyBackbone(),8)
        lo=model(torch.randn(4,3,8,8));so=model(torch.randn(4,3,8,8))
        teacher_projection=torch.randn(4,128,requires_grad=True)
        bank=fit_bank(torch.nn.functional.normalize(torch.randn(40,8),dim=-1),torch.tensor([0]*20+[1]*20),10,3,8)
        q=torch.softmax(torch.randn(4,10),-1)
        target=dict(q=q,a_target=q,a=torch.tensor([1,0,0,0],dtype=torch.bool),b=torch.tensor([0,1,0,0],dtype=torch.bool),
                    c=torch.tensor([0,0,1,1],dtype=torch.bool),weight=torch.ones(4),score=torch.ones(4)*.8)
        loss,terms=objective(model,lo,so,teacher_projection,torch.tensor([0,1,0,1]),target,bank,1,self.a)
        loss.backward()
        self.assertTrue(all(float(t.detach())>0 for t in terms.values()))
        self.assertIsNone(teacher_projection.grad)
        self.assertGreater(float(model.predictor[0].weight.grad.norm()),0)

    def test_full_round_checkpoint_resume_matches_uninterrupted(self):
        class ManyPictures(Dataset):
            def __init__(self):self.images=Pictures(10).images
            def __len__(self):return 6000
            def __getitem__(self,i):
                if i>=len(self):raise IndexError(i)
                return self.images[i%10],i%10
        class EvalPictures(Dataset):
            def __len__(self):return 10
            def __getitem__(self,i):return torch.full((3,32,32),i/10),i
        a=copy.deepcopy(self.a);a.num_clients=20;a.num_online_clients=1
        a.batch_size_local_labeled_fixmatch=128;a.tm_main_rounds=2;a.max_rounds=2;a.tm_start=0;a.tm_ramp=1
        a.sample_seed=7
        dataset=ManyPictures();test=EvalPictures();original_cwd=os.getcwd()
        def tiny(device):return Learner(TinyBackbone(),8).to(device)
        with tempfile.TemporaryDirectory() as tmp,patch('trusted_multi_runner.make_model',tiny),contextlib.redirect_stdout(io.StringIO()):
            try:
                os.chdir(tmp)
                a.run_id='complete';run(a,train_data=dataset,test_data=test,device=torch.device('cpu'))
                full=torch.load(Path('results/CIFAR10/runs/complete_a0.1/checkpoint.pt'),weights_only=False)
                a.run_id='interrupted'
                def interrupt(value,path):
                    atomic_save(value,path)
                    if value['round']==1:raise RuntimeError('simulated interruption')
                with patch('trusted_multi_runner.atomic_save',interrupt):
                    with self.assertRaisesRegex(RuntimeError,'simulated interruption'):
                        run(a,train_data=dataset,test_data=test,device=torch.device('cpu'))
                a.resume=str(Path('results/CIFAR10/runs/interrupted_a0.1/checkpoint.pt').resolve())
                run(a,train_data=dataset,test_data=test,device=torch.device('cpu'))
                resumed=torch.load(a.resume,weights_only=False)
                for k,v in full['student'].items():torch.testing.assert_close(v,resumed['student'][k],rtol=0,atol=0)
                for k,v in full['teacher'].items():torch.testing.assert_close(v,resumed['teacher'][k],rtol=0,atol=0)
                self.assertEqual(full['accuracy'],resumed['accuracy'])
                self.assertEqual(full['config']['partition_sha256'],resumed['config']['partition_sha256'])
            finally:os.chdir(original_cwd)


if __name__=='__main__':unittest.main()
