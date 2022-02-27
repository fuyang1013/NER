import torch


def sparse_label_ids(label_dense_matrices, heads):
    """
    将输入的【稠密标签矩阵】转化成【多头标签矩阵】,便于后面和globalpointer输出做对比

    params:
        label_dense_matrices(IntTensor): 稠密的标签
        heads(int): head数量
    return:
        label_multihead_matrices: 多头矩阵
    """

    label_heads = [torch.unsqueeze(label_dense_matrices, 1) == h+1 for h in range(heads)]
    return torch.cat(label_heads, dim=1).int()
