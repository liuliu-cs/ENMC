import numpy as np
import hnswlib
# from sitq import Sitq
from sitq import Mips
import faiss

n = 32317
d = 1024
num_candidates = 5

Wb = np.random.rand(n, d+1)
X = np.random.rand(10, d+1)

# print('Using KNN search')
# G = hnswlib.Index(space='ip', dim=d+1)
# G.init_index(max_elements=n, ef_construction=2000, M=1024)
# indices = np.arange(n)
# G.add_items(Wb, indices)
# G.set_ef(2000)

# print('Graph construction done, Wb: {}'.format(Wb.shape))
# # KNN search for indices
# I, _ = G.knn_query(x, k=num_candidates)
# print(I)

# sitq = Sitq(signature_size=8)
# sitq.fit(Wb)
mips = Mips(signature_size=16)
mips.fit(Wb)

for x in X:
    I, scores = mips.search(x, limit=10, distance=1)
    print(I)
    z = np.matmul(x, Wb.transpose())
    print(np.argmax(z))
# full dot-product search
# z = np.matmul(x, Wb.transpose())
# print(z.shape)
# print(np.argmax(z, axis=1))

# Create sample dataset
items = np.random.rand(32317, 1024)
queries = np.random.rand(10, 1024)

mips = Mips(signature_size=8)

# Learn lookup table and parameters for search
mips.fit(items)

# Find items which are likely to maximize inner product against query
for query in queries:
    item_indexes, scores = mips.search(query, limit=10, distance=1)
    print(item_indexes)
    output = np.matmul(query, items.transpose())
    top = np.argmax(output)
    if top in item_indexes: print(top)