# -*- coding: utf-8 -*-
"""
联邦场景下按「类别 -> 客户端」分配样本索引。

- clients_indices_homo：IID，各类样本按顺序均分给各客户端。
- clients_indices / clients_indices_unlabel*：对 (索引, 标签) 做 Dirichlet 采样，
  使各客户端类别比例随机倾斜，模拟 Non-IID；再 balance 切成 num_clients 份。

与 SAGE.py 的关系：alpha>0 时对「有标池」「无标池」分别调用 clients_indices。
"""
import math
import functools
import numpy as np

def clients_indices_homo(list_label2indices: list, num_classes: int, num_clients: int):
    """同类样本按位置切片均分，合并后得到每个客户端的索引数组列表。"""
    partition_list = []
    for index_class in range(num_classes):
        class_partition_list = []
        class_list_length = len(list_label2indices[index_class])
        for client in range(num_clients):
            class_partition = list_label2indices[index_class][math.floor(client/num_clients * class_list_length):math.floor((client+1) / num_clients * class_list_length)]

            class_partition_list.append(class_partition)
        partition_list.append(class_partition_list)

    list_label2indices = []
    for client in range(num_clients):
        client_partition_list = []
        for class_idx in range(num_classes):
            client_partition_list.extend(partition_list[class_idx][client])
        list_label2indices.append(np.array(client_partition_list))


    return list_label2indices



def clients_indices(list_label2indices: list, num_classes: int, num_clients: int, non_iid_alpha: float, seed=None):
    """Dirichlet Non-IID 划分（有标侧与 SAGE 主流程无标侧共用同一套逻辑结构）。"""
    indices2targets = []


    for label, indices in enumerate(list_label2indices):
        for idx in indices:
            indices2targets.append((idx, label))


    batch_indices = build_non_iid_by_dirichlet(seed=seed,
                                               indices2targets=indices2targets,
                                               non_iid_alpha=non_iid_alpha,
                                               num_classes=num_classes,
                                               num_indices=len(indices2targets),
                                               n_workers=num_clients)

    indices_dirichlet = functools.reduce(lambda x, y: x + y, batch_indices)

    list_client2indices = partition_balance(indices_dirichlet, num_clients)

    return list_client2indices


def partition_balance(idxs, num_split: int):
    """将展平后的索引序列尽量均匀切成 num_split 段（余数前若干段多 1 个）。"""

    num_per_part, r = len(idxs) // num_split, len(idxs) % num_split
    parts = []
    i, r_used = 0, 0


    while i < len(idxs):
        if r_used < r:

            parts.append(idxs[i:(i + num_per_part + 1)])
            i += num_per_part + 1
            r_used += 1
        else:

            parts.append(idxs[i:(i + num_per_part)])
            i += num_per_part

    return parts


def build_non_iid_by_dirichlet(
    seed, indices2targets, non_iid_alpha, num_classes, num_indices, n_workers):
    """
    核心：对每个辅助子批、每一类，用 Dirichlet(non_iid_alpha) 抽样比例，
    把该类索引划分给多个「工作节点」，循环直到每份至少有一定最小量。
    """
    random_state = np.random.RandomState(seed)
    n_auxi_workers = 2
    assert n_auxi_workers <= n_workers

    # random shuffle targets indices.
    random_state.shuffle(indices2targets)

    # partition indices.
    from_index = 0
    splitted_targets = []

    num_splits = math.ceil(n_workers / n_auxi_workers)

    split_n_workers = [
        n_auxi_workers
        if idx < num_splits - 1
        else n_workers - n_auxi_workers * (num_splits - 1)
        for idx in range(num_splits)
    ]
    split_ratios = [_n_workers / n_workers for _n_workers in split_n_workers]
    for idx, ratio in enumerate(split_ratios):
        to_index = from_index + int(n_auxi_workers / n_workers * num_indices)
        splitted_targets.append(
            indices2targets[
                from_index: (num_indices if idx == num_splits - 1 else to_index)
            ]
        )
        from_index = to_index

    #
    idx_batch = []
    for _targets in splitted_targets:
        # rebuild _targets.
        _targets = np.array(_targets)
        _targets_size = len(_targets)

        # use auxi_workers for this subset targets.
        _n_workers = min(n_auxi_workers, n_workers)
        #n_workers=10
        n_workers = n_workers - n_auxi_workers

        # get the corresponding idx_batch.
        min_size = 0
        _idx_batch = None
        while min_size < int(0.50 * _targets_size / _n_workers):
            _idx_batch = [[] for _ in range(_n_workers)]
            for _class in range(num_classes):
                # get the corresponding indices in the original 'targets' list.
                idx_class = np.where(_targets[:, 1] == _class)[0]
                idx_class = _targets[idx_class, 0]

                # sampling.
                try:
                    proportions = random_state.dirichlet(
                        np.repeat(non_iid_alpha, _n_workers)
                    )
                    # balance
                    proportions = np.array(
                        [
                            p * (len(idx_j) < _targets_size / _n_workers)
                            for p, idx_j in zip(proportions, _idx_batch)
                        ]
                    )
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_class)).astype(int)[
                        :-1
                    ]
                    _idx_batch = [
                        idx_j + idx.tolist()
                        for idx_j, idx in zip(
                            _idx_batch, np.split(idx_class, proportions)
                        )
                    ]
                    sizes = [len(idx_j) for idx_j in _idx_batch]
                    min_size = min([_size for _size in sizes])
                except ZeroDivisionError:
                    pass
        if _idx_batch is not None:
            idx_batch += _idx_batch

    return idx_batch



def clients_indices_unlabel(list_label2indices: list, num_classes: int, num_clients: int, non_iid_alpha: float, seed=None):
    """无标池的 Dirichlet 划分；内部 n_auxi_workers=9 等与 clients_indices 略有不同。"""
    indices2targets = []
    for label, indices in enumerate(list_label2indices):
        for idx in indices:
            indices2targets.append((idx, label))

    batch_indices = build_non_iid_by_dirichlet_unlabel(seed=seed,
                                               indices2targets=indices2targets,
                                               non_iid_alpha=non_iid_alpha,
                                               num_classes=num_classes,
                                               num_indices=len(indices2targets),
                                               n_workers=num_clients)
    indices_dirichlet = functools.reduce(lambda x, y: x + y, batch_indices)
    list_client2indices = partition_balance_unlabel(indices_dirichlet, num_clients)

    return list_client2indices


def partition_balance_unlabel(idxs, num_split: int):
    """与 partition_balance 相同思想，用于 unlabel 分支输出。"""

    num_per_part, r = len(idxs) // num_split, len(idxs) % num_split
    parts = []
    i, r_used = 0, 0
    while i < len(idxs):
        if r_used < r:
            parts.append(idxs[i:(i + num_per_part + 1)])
            i += num_per_part + 1
            r_used += 1
        else:
            parts.append(idxs[i:(i + num_per_part)])
            i += num_per_part

    return parts


def build_non_iid_by_dirichlet_unlabel(
    seed, indices2targets, non_iid_alpha, num_classes, num_indices, n_workers
):
    """无标版 Dirichlet 切块；n_auxi_workers 默认 9。"""
    random_state = np.random.RandomState(seed)
    n_auxi_workers = 9
    assert n_auxi_workers <= n_workers

    # random shuffle targets indices.
    random_state.shuffle(indices2targets)

    # partition indices.
    from_index = 0
    splitted_targets = []

    num_splits = math.ceil(n_workers / n_auxi_workers)

    split_n_workers = [
        n_auxi_workers
        if idx < num_splits - 1
        else n_workers - n_auxi_workers * (num_splits - 1)
        for idx in range(num_splits)
    ]
    split_ratios = [_n_workers / n_workers for _n_workers in split_n_workers]
    for idx, ratio in enumerate(split_ratios):
        to_index = from_index + int(n_auxi_workers / n_workers * num_indices)
        splitted_targets.append(
            indices2targets[
                from_index: (num_indices if idx == num_splits - 1 else to_index)
            ]
        )
        from_index = to_index

    #
    idx_batch = []
    for _targets in splitted_targets:
        # rebuild _targets.
        _targets = np.array(_targets)
        _targets_size = len(_targets)

        # use auxi_workers for this subset targets.
        _n_workers = min(n_auxi_workers, n_workers)
        #n_workers=10
        n_workers = n_workers - n_auxi_workers

        # get the corresponding idx_batch.
        min_size = 0
        _idx_batch = None
        while min_size < int(0.50 * _targets_size / _n_workers):
            _idx_batch = [[] for _ in range(_n_workers)]
            for _class in range(num_classes):
                # get the corresponding indices in the original 'targets' list.
                idx_class = np.where(_targets[:, 1] == _class)[0]
                idx_class = _targets[idx_class, 0]

                # sampling.
                try:
                    proportions = random_state.dirichlet(
                        np.repeat(non_iid_alpha, _n_workers)
                    )
                    # balance
                    proportions = np.array(
                        [
                            p * (len(idx_j) < _targets_size / _n_workers)
                            for p, idx_j in zip(proportions, _idx_batch)
                        ]
                    )
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_class)).astype(int)[
                        :-1
                    ]
                    _idx_batch = [
                        idx_j + idx.tolist()
                        for idx_j, idx in zip(
                            _idx_batch, np.split(idx_class, proportions)
                        )
                    ]
                    sizes = [len(idx_j) for idx_j in _idx_batch]
                    min_size = min([_size for _size in sizes])
                except ZeroDivisionError:
                    pass
        if _idx_batch is not None:
            idx_batch += _idx_batch

    return idx_batch



def clients_indices_unlabel1(list_label2indices: list, num_classes: int, num_clients: int, non_iid_alpha: float, seed=None):
    """另一组超参（n_auxi_workers=10）的无标划分变体，供对比实验保留。"""
    indices2targets = []
    for label, indices in enumerate(list_label2indices):
        for idx in indices:
            indices2targets.append((idx, label))

    batch_indices = build_non_iid_by_dirichlet_unlabel1(seed=seed,
                                               indices2targets=indices2targets,
                                               non_iid_alpha=non_iid_alpha,
                                               num_classes=num_classes,
                                               num_indices=len(indices2targets),
                                               n_workers=num_clients)
    indices_dirichlet = functools.reduce(lambda x, y: x + y, batch_indices)
    list_client2indices = partition_balance_unlabel1(indices_dirichlet, num_clients)

    return list_client2indices


def partition_balance_unlabel1(idxs, num_split: int):
    """unlabel1 变体使用的均匀切分。"""
    num_per_part, r = len(idxs) // num_split, len(idxs) % num_split
    parts = []
    i, r_used = 0, 0
    while i < len(idxs):
        if r_used < r:
            parts.append(idxs[i:(i + num_per_part + 1)])
            i += num_per_part + 1
            r_used += 1
        else:
            parts.append(idxs[i:(i + num_per_part)])
            i += num_per_part

    return parts


def build_non_iid_by_dirichlet_unlabel1(
    seed, indices2targets, non_iid_alpha, num_classes, num_indices, n_workers
):
    """与 build_non_iid_by_dirichlet_unlabel 类似，辅助 worker 数改为 10。"""
    random_state = np.random.RandomState(seed)
    n_auxi_workers = 10
    assert n_auxi_workers <= n_workers

    # random shuffle targets indices.
    random_state.shuffle(indices2targets)

    # partition indices.
    from_index = 0
    splitted_targets = []

    num_splits = math.ceil(n_workers / n_auxi_workers)

    split_n_workers = [
        n_auxi_workers
        if idx < num_splits - 1
        else n_workers - n_auxi_workers * (num_splits - 1)
        for idx in range(num_splits)
    ]
    split_ratios = [_n_workers / n_workers for _n_workers in split_n_workers]
    for idx, ratio in enumerate(split_ratios):
        to_index = from_index + int(n_auxi_workers / n_workers * num_indices)
        splitted_targets.append(
            indices2targets[
                from_index: (num_indices if idx == num_splits - 1 else to_index)
            ]
        )
        from_index = to_index

    #
    idx_batch = []
    for _targets in splitted_targets:
        # rebuild _targets.
        _targets = np.array(_targets)
        _targets_size = len(_targets)

        # use auxi_workers for this subset targets.
        _n_workers = min(n_auxi_workers, n_workers)
        #n_workers=10
        n_workers = n_workers - n_auxi_workers

        # get the corresponding idx_batch.
        min_size = 0
        _idx_batch = None
        while min_size < int(0.50 * _targets_size / _n_workers):
            _idx_batch = [[] for _ in range(_n_workers)]
            for _class in range(num_classes):
                # get the corresponding indices in the original 'targets' list.
                idx_class = np.where(_targets[:, 1] == _class)[0]
                idx_class = _targets[idx_class, 0]

                # sampling.
                try:
                    proportions = random_state.dirichlet(
                        np.repeat(non_iid_alpha, _n_workers)
                    )
                    # balance
                    proportions = np.array(
                        [
                            p * (len(idx_j) < _targets_size / _n_workers)
                            for p, idx_j in zip(proportions, _idx_batch)
                        ]
                    )
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_class)).astype(int)[
                        :-1
                    ]
                    _idx_batch = [
                        idx_j + idx.tolist()
                        for idx_j, idx in zip(
                            _idx_batch, np.split(idx_class, proportions)
                        )
                    ]
                    sizes = [len(idx_j) for idx_j in _idx_batch]
                    min_size = min([_size for _size in sizes])
                except ZeroDivisionError:
                    pass
        if _idx_batch is not None:
            idx_batch += _idx_batch

    return idx_batch
