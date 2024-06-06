import numpy as np
from Smallest_k import IP
import warnings
X = {}
y = {}
path = "./data/emo_bert/"
X["train"] = np.load(path + "train_feature_save.npy",allow_pickle=True )[:9025]
y["train"] = np.load(path + "train_label_save.npy")[:9025].squeeze()
X["dev"] = np.load(path + "test_feature_save.npy")[:1003]
y["dev"] = np.load(path + "test_label_save.npy")[:1003].squeeze()

print(X["train"].shape, y["train"].shape, X["dev"].shape, y["dev"].shape)
thresh = 0.5
l2 = 10
IP(X, y, l2, "emo_b", thresh)


import numpy as np
from Smallest_k import IP
import warnings
X = {}
y = {}
path = "./data/emo/"
X["train"] = np.load(path + "X_train.npy")
y["train"] = np.load(path + "y_train.npy")
X["dev"] = np.load(path + "X_dev.npy")
y["dev"] = np.load(path + "y_dev.npy")

print(X["train"].shape, y["train"].shape, X["dev"].shape, y["dev"].shape)
thresh = 0.5
l2 = 10 # default 0.77
l2 = 100 # 0.71
#l2 = 1000 #0.63
IP(X, y, l2, "emo", thresh)