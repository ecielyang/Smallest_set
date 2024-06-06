import numpy as np
from Smallest_k import IP
from recursive import IP_iterative
import warnings
X = {}
y = {}
path = "./data/SST_bert/"
X["train"] = np.load(path + "train_feature_save.npy",allow_pickle=True )
y["train"] = np.load(path + "train_label_save.npy").squeeze()
X["dev"] = np.load(path + "test_feature_save.npy")
y["dev"] = np.load(path + "test_label_save.npy").squeeze()

print(X["train"].shape, y["train"].shape, X["dev"].shape, y["dev"].shape)
thresh = 0.5
l2 = 10

# Algorithm 1
IP(X, y, l2, "SST_b", thresh)

# Algorithm2
IP_iterative(X, y, l2, "SST_b", thresh)

