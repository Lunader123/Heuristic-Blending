import numpy as np
def lambda_constant():
    """常数函数 λ(τ)"""
    alpha = 0.8
    return alpha

def lambda_sigmoid(h, sigma = 0.1, alpha = 0.8):
    """Sigmoid函数 λ(τ)"""
    avg_h = np.mean(h)
    t = len(h)
    return alpha * np.exp(-avg_h / sigma) / (1 + np.exp(-avg_h / sigma))

def lambda_rank(h,  alpha = 0.8):
    """排序函数 λ(τ)"""
    h_ranks = np.argsort(h)  # 根据每个h的大小进行排序
    n = len(h)
    avg_ranked_h = np.mean(h[h_ranks[:n]])
    return alpha * avg_ranked_h