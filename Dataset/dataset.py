# -*- coding: utf-8 -*-
"""
联邦半监督场景下的数据工具：按类别整理索引、划分有标/无标、封装为 PyTorch Dataset。

与 SAGE.py 的配合方式：
- classify_label + partition_train 得到全数据的有标/无标索引列表；
- sample_dirichlet 将索引划分到各客户端；
- Indices2Dataset_* 按客户端索引列表惰性加载子集，并施加 FixMatch 增广。
"""
import numpy as np
from torch.utils.data.dataset import Dataset
import copy
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms

from .randaugment import RandAugmentMC
from Dataset.sample_dirichlet import clients_indices, clients_indices_unlabel

import time



def classify_label(dataset, num_classes: int):
    """返回 list[类 id] -> 属于该类的样本下标列表。"""
    list1 = [[] for _ in range(num_classes)]
    for idx, datum in enumerate(dataset):
        list1[datum[1]].append(idx)
    return list1



def show_clients_data_distribution(dataset, clients_indices_labeled, clients_indices_unlabeled, num_classes):
    """调试：打印每个客户端、每个类别上的有标/无标样本数量。"""
    dict_per_client_labeled = []

    for client, indices in enumerate(zip(clients_indices_labeled, clients_indices_unlabeled)):
        nums_data_labeled = [0 for _ in range(num_classes)]
        nums_data_unlabeled = [0 for _ in range(num_classes)]
        idx_labeled, idx_unlabeled = indices
        for idx in idx_labeled:
            label = dataset[idx][1]
            nums_data_labeled[label] += 1
        dict_per_client_labeled.append(nums_data_labeled)
        for idx in idx_unlabeled:
            label = dataset[idx][1]
            nums_data_unlabeled[label] += 1
        print(f'client {client} labeled number per class : {nums_data_labeled}')
        print(f'client {client} unlabeled number per class  : {nums_data_unlabeled}')
    return dict_per_client_labeled


def partition_train(list_label2indices: list, ipc):
    """
    对每一类：随机打乱后前 ipc 个作为有标注，其余为无标注。
    注意：SAGE.py 里传入的 args.num_labeled 即此处的 ipc，表示「每类」有标数。
    """
    list_label2indices_labeled = []
    list_label2indices_unlabeled = []

    for indices in list_label2indices:

        idx_shuffle = np.random.permutation(indices)

        list_label2indices_labeled.append(idx_shuffle[:ipc])
        list_label2indices_unlabeled.append(idx_shuffle[ipc:])
    return list_label2indices_labeled, list_label2indices_unlabeled


def compute_clients_labeled_data_distribution(dataset, clients_indices_labeled, num_classes):
    """统计给定索引列表上各类样本数量（供其它采样脚本使用）。"""
    dict_per_client_labeled = []
    nums_data_labeled = [0 for _ in range(num_classes)]
    for idx in clients_indices_labeled:
        label = dataset[idx][1]
        nums_data_labeled[label] += 1
    dict_per_client_labeled.append(nums_data_labeled)
    return dict_per_client_labeled


def partition_train_teach(list_label2indices: list, ipc, seed=None):
    """教师模型等场景：每类取前 ipc 个索引（与 partition_train 不同，未拆无标池）。"""
    random_state = np.random.RandomState(0)
    list_label2indices_teach = []

    for indices in list_label2indices:
        random_state.shuffle(indices)
        list_label2indices_teach.append(indices[:ipc])

    return list_label2indices_teach


def partition_unlabel(list_label2indices: list, num_data_train: int):
    """每类取 num_data_train//100 个无标样本（本主流程未用）。"""
    random_state = np.random.RandomState(0)
    list_label2indices_unlabel = []

    for indices in list_label2indices:
        random_state.shuffle(indices)
        list_label2indices_unlabel.append(indices[:num_data_train // 100])
    return list_label2indices_unlabel


def label_indices2indices(list_label2indices):
    """将「按类分的索引列表」展平为一个列表。"""
    indices_res = []
    for indices in list_label2indices:
        indices_res.extend(indices)

    return indices_res




class Indices2Dataset_labeled(Dataset):
    """有标注分支：load(indices) 后仅迭代这些样本；弱增广 + CIFAR 归一化。"""

    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = None
        # 预先创建 transform，避免每次 __getitem__ 都重新创建（极慢！）
        self.label_trans = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
        ])

    def load(self, indices: list):
        """绑定当前客户端的有标索引；列表重复多次以减少 DataLoader 重建迭代器开销。"""
        self.indices = indices
        self.client_dataset = [self.dataset[i] for i in indices]
        self.client_dataset_original_len = len(self.client_dataset)  # 保存原始长度
        self.client_dataset *= 2000

    def __getitem__(self, idx):
        image, label = self.client_dataset[idx]
        image = self.label_trans(image)
        return image, label

    def __len__(self):
        return len(self.client_dataset)


class Indices2Dataset_unlabeled_fixmatch(Dataset):
    """无标注分支：返回 (弱增广, 强增广, 真实标签仅用于日志/分析)。"""

    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = None
        # 预先创建 transforms，避免每次 __getitem__ 重复创建
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode='reflect'),
        ])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode='reflect'),
            RandAugmentMC(n=2, m=10),
        ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))
        ])

    def load(self, indices: list):
        """绑定当前客户端无标索引；内部列表重复若干倍以摊薄迭代器开销。"""
        self.indices = indices
        self.client_dataset = [self.dataset[i] for i in self.indices]
        self.client_dataset_len = len(self.client_dataset)
        self.client_dataset *= 50

    def __getitem__(self, idx):
        image, label = self.client_dataset[idx]
        weak = self.weak(image)
        strong = self.strong(image)
        return self.normalize(weak), self.normalize(strong), label

    def __len__(self):
        return self.client_dataset_len



def list_IID_clients(list_label2indices_labeled, list_label2indices_unlabeled, num_classes, num_clients):
    """将每类有标/无标均匀切成 num_clients 份并重组为按客户端的索引列表（本主流程未用）。"""

    labeled_per_client = int(len(list_label2indices_labeled[0]) / num_clients)
    unlabeled_per_client = int(len(list_label2indices_unlabeled[0]) / num_clients)

    random_state = np.random.RandomState(0)
    for class_id, data_all in enumerate(zip(list_label2indices_labeled, list_label2indices_unlabeled)):
        labeled_data, unlabeled_data = data_all
        list_label2indices_labeled[class_id] = [labeled_data[i:i + labeled_per_client] for i in range(0, len(labeled_data), labeled_per_client)]
        list_label2indices_unlabeled[class_id] = [unlabeled_data[i:i + unlabeled_per_client] for i in range(0, len(unlabeled_data), unlabeled_per_client)]

    list_clients_labeled = [[] for i in range(num_clients)]
    list_clients_unlabeled = [[] for i in range(num_clients)]
    for client_id in range(num_clients):
        for class_id in range(num_classes):
            list_clients_labeled[client_id].extend(list(list_label2indices_labeled[class_id][client_id]))
            list_clients_unlabeled[client_id].extend(list(list_label2indices_unlabeled[class_id][client_id]))

    return list_clients_labeled, list_clients_unlabeled




def sampling_labeled_data_non_iid(args, data_local_training, list_label2indices_labeled, num_labeled_client, alpha, seed=0):
    """实验性：更复杂的有标 Non-IID 划分（SAGE 主脚本未调用）。"""
    list_choose_labeled = []
    list_choose_labeled_client1 = []
    list_rest_label2indices_labeled = []
    random_state = np.random.RandomState(seed)


    list_choose_labeled_non_iid = clients_indices(list_label2indices=list_label2indices_labeled,
                                                  num_classes=args.num_classes,
                                                  num_clients=2,
                                                  non_iid_alpha=alpha,
                                                  seed=seed)


    client1_sampling = compute_clients_labeled_data_distribution(data_local_training,
                                                                 list_choose_labeled_non_iid[0],
                                                                 args.num_classes)
    client1_sampling = client1_sampling[0]
    for class_idx, list_index in enumerate(list_label2indices_labeled):

        new_data = set(random_state.choice(list_index, client1_sampling[class_idx], replace=False))
        list_new_data = list(new_data)
        list_choose_labeled_client1.extend(list_new_data)
        list_index = list(set(list_index) - new_data)
        list_rest_label2indices_labeled.append(list_index)

    list_choose_labeled.append(list_choose_labeled_client1)


    list_choose_labeled_rest_client = clients_indices_unlabel(list_label2indices=list_rest_label2indices_labeled,
                                                        num_classes=args.num_classes,
                                                        num_clients=(num_labeled_client-1),
                                                        non_iid_alpha=alpha,
                                                        seed=10)
    list_choose_labeled.extend(list_choose_labeled_rest_client)

    return list_choose_labeled


def sampling_unlabeled_data_non_iid(args, list_label2indices_unlabeled,
                                    num_unlabeled_client, alpha, seed=0):
    """实验性：无标数据分块 + Dirichlet（SAGE 主脚本未调用）。"""
    list_choose_unlabeled = []
    list_unlabeled_part1 = []
    list_unlabeled_part2 = []
    random_state = np.random.RandomState(0)
    class_sampling = [2000] * 10
    for class_idx, list_index in enumerate(list_label2indices_unlabeled):
        new_data = set(random_state.choice(list_index, class_sampling[class_idx], replace=False))
        list_new_data = list(new_data)
        list_unlabeled_part1.append(list_new_data)
        list_index = list(set(list_index) - new_data)
        list_unlabeled_part2.append(list_index)


    list_client_part1 =clients_indices_unlabel(list_label2indices=list_unlabeled_part1,
                                        num_classes=args.num_classes, num_clients=9,
                                        non_iid_alpha=alpha, seed=1000)
    list_client_part2 = clients_indices_unlabel(list_label2indices=list_unlabeled_part2, num_classes=args.num_classes, num_clients=10,
                                        non_iid_alpha=alpha, seed=1000)
    list_choose_unlabeled.append([])
    list_choose_unlabeled.extend(list_client_part1)
    list_choose_unlabeled.extend(list_client_part2)

    return list_choose_unlabeled

