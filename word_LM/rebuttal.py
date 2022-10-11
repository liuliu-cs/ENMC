# pip install hnswlib
# pip install sitq
# install taco (https://github.com/tensor-compiler/taco/)
#   1) cmake -DCMAKE_BUILD_TYPE=Release -DPYTHON=ON ..
#   2) export PYTHONPATH=<taco-directory>/build/lib:$PYTHONPATH

import torch
import numpy as np
import hnswlib
# import pytaco as pt
# from pytaco import dense, compressed
import time

# M = np.load('classifier_mask.npy')
# ratio = M.sum() / M.size
# print('Mask: {}, NNZ ratio: {:.2f}, Saving: {:.2f}'.format(M.shape, ratio, 1 / ratio))
mbs = 1
# M = M[:mbs, :]
# mbs = M.shape[0]
# n = M.shape[1]
n = 33278
d = 1500
k = int(d * 0.25)
W = np.random.randn(n, d)
X = np.random.randn(mbs, d)

################### Numpy ########################

num_runs = 1000
np.matmul(X, W.transpose())
duration_np = 0
for _ in range(num_runs):
    start = time.time()
    np.matmul(X, W.transpose())
    duration_np += time.time() - start
print('Numpy: {:.3f} ms'.format(duration_np * 1e3 / num_runs))

# ################ ENMC (w/o BLAS) ######################
M = torch.randn(n, mbs)
dim = 0
nnz = 256
_, indices = torch.topk(M.abs().contiguous(), nnz, dim=dim, sorted=False)
M = torch.zeros(n, mbs).scatter_(dim, indices, 1.0)
print('M: {}, {:.2f}; X: {}; W: {}; IL {}'.format(M.shape, M.sum() / M.numel(), X.shape, W.shape, indices.shape))

# W_indices = np.nonzero(M.numpy())
W_indices = np.squeeze(indices)
duration_dre = 0
for _ in range(num_runs):
    W_dre = W[W_indices]
    start = time.time()
    # print(X.shape, W_dre.shape)
    np.matmul(X, W_dre.transpose())
    duration_dre += time.time() - start
print('ENMC: {:.3f} ms; speedup: {:.2f}'.format(duration_dre * 1e3 / num_runs, duration_np / duration_dre))

################ KNN (+SVD for LM tasks) ######################
# offline construction
Graph = hnswlib.Index(space='l2', dim=d)
Graph.init_index(max_elements=n, ef_construction=200, M=16)
labels = np.arange(n)
Graph.add_items(W, labels)
Graph.set_ef(50)
# online search
h = X
print('KNN, h: ', h.shape)
knn_indices, _ = Graph.knn_query(h, k=1)
duration_knn = 0
for _ in range(num_runs):
    start = time.time()
    knn_indices, _ = Graph.knn_query(h, k=20)
    duration_knn += time.time() - start 
print('KNN search: {:.3f} ms; speedup: {:.2f}'.format(duration_knn * 1e3 / num_runs, duration_np / duration_knn))



# ################### TACO (W/BLAS) ########################
# W_t_pt = pt.from_array(W.transpose())
# X_pt = pt.from_array(X)


# num_runs = 1000
# duration_pt = 0
# for _ in range(num_runs):
#     start = time.time()
#     Z_d = pt.matmul(X_pt, W_t_pt)
#     duration_pt += time.time() - start
# print('TACO: {:.3f} ms'.format(duration_pt * 1e3 / num_runs))


# # ################ ENMC (w/ BLAS) ######################
# dcsr = pt.format([compressed, compressed])
# rm = pt.format([dense, dense])
# cm = pt.format([dense, dense], [1, 0])
# M_dcsr = pt.remove_explicit_zeros(M.numpy().transpose(), dcsr)
# Z_pt = pt.tensor(M_dcsr.shape, dcsr)
# print('Shape Z {} = M {} (.) X {} (x) W_t {}'.format(Z_pt.shape, M_dcsr.shape, X_pt.shape, W_t_pt.shape))
# i, j, k = pt.get_index_vars(3)
# Z_pt[i, j] = M_dcsr[i, j] * X_pt[i, k] * W_t_pt[k, j]
# Z_pt.compile()
# Z_pt.assemble()
# duration_dre = 0
# for _ in range(num_runs):
#     start = time.time()
#     Z_pt.compute()
#     duration_dre += time.time() - start
# print('ENMC: {:.3f} ms; speedup: {:.2f}'.format(duration_dre * 1e3 / num_runs, duration_pt/ duration_dre))