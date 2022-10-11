import numpy as np
import matplotlib.pyplot as plt 

# m = np.load('classifier_mask.npy')
# print(m.shape)
# # m = np.reshape(m, (10, 35, -1))
# # print(m.shape)
# # print(m[:,0,:50])
# # y = np.sum(m[:,0,:], axis=(0, 1))
# # print(y)

# # plt.matshow(m[:,0,:])
# plt.matshow(m)

# plt.savefig('classifier_mask')

m = np.load("knn_mask.npy")

plt.matshow(m)
plt.savefig('knn_mask')