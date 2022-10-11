import math
import numpy as np
import torch

def fit(n, d, eps=0.5, scale=None, dist='binary'):
    # n samples with original dim d
    if scale:
        k = int(d * scale)
    else:
        k = calc_min_dim(n_samples=n, eps=eps)
    
    if k <= 0:
        raise ValueError("Target dimension is invalid, got %d" % k)
    elif k >= d:
        print("Target dimension is greater than original dimension")

    print("# of samples: {}, target dim: {}, original dim: {}; epsilon: {}".format(
            n, k, d, eps))

    if dist == 'binary':
        np_sparse_matrix = np.random.binomial(1, 0.5, size=(d, k)) * 2 - 1
    else:
        # density controlled by s
        s = 3
        np_sparse_matrix = np.random.binomial(1, float(1 / s), size=(d, k)) / math.sqrt(s)
        signs = np.random.binomial(1, 0.5, size=(d, k)) * 2 - 1
        np_sparse_matrix = np_sparse_matrix * signs
    return torch.from_numpy(np_sparse_matrix).to(torch.float)

def calc_min_dim(n_samples, eps):
    if eps <= 0.0 or eps >= 1:
        raise ValueError(
            "The eps for JL lemma should in [0, 1], got %r" % eps)
    if n_samples <=0:
        raise ValueError(
            "The number of samples should be greater than zero, got %r" % n_samples)
    denominator = (eps ** 2 / 2) - (eps ** 3 / 3)
    return int(4 * np.log(n_samples) / denominator)

# Sparse random projection
# return results in low dimensional space
def srp(X, proj_matrix):
    assert X.size(-1) == proj_matrix.size(0)
    return torch.matmul(X, proj_matrix)

def main():
    X = torch.randn(16, 32, 3, 3)
    proj = fit(16, 32*3*3, scale=float(1/3))
    print(proj.shape)
    print(proj)

if __name__ == '__main__':
    main()