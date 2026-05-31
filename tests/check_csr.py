import scipy.sparse as sp
csr = sp.load_npz("data/processed/train_csr.npz")
print(csr.data.min(), csr.data.max(), csr.data.mean())