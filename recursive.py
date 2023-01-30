import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.linear_model import LogisticRegression

def concatenate(X):
    return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)


def error(X, y, model):
    return model.predict_proba(X)[:, 1] - y


def prediction(X, model):
    return model.predict_proba(X)[:, 1]


def gradient_(X, y, model, C):
    F = concatenate(X)
    # return F * error(X, y, model)[:, None] + C * w / X.shape[0]
    return F * error(X, y, model)[:, None]


def hessian_(X, model, C):
    probs = prediction(X, model)
    F = concatenate(X)
    H = F.T @ np.diag(probs * (1 - probs)) @ F / F.shape[0] + C * np.eye(
        F.shape[1]) / X.shape[0]
    return H


def inverse_hessian(H):
    return np.linalg.inv(H)

def Remove(X, y, k, scores, test_idx, pred, thresh):
    # print("test_idx", test_idx)
    # print("old")
    # print(pred[test_idx])

    if pred[test_idx] > thresh:
        top_k_index = scores[test_idx].argsort()[-k:]
    else:
        top_k_index = scores[test_idx].argsort()[:k]

    X_r = X["train"][top_k_index]
    y_r = y["train"][top_k_index]
    X_l = np.delete(X["train"], top_k_index, axis=0)
    y_l = np.delete(y["train"], top_k_index, axis=0)

    prediction = -np.sum(scores[test_idx][top_k_index])
    # print("prediction", prediction)

    return X_l, y_l, prediction, X_r, y_r


# In[3]:


def new_train(X, y, l2, k, dev_index, scores, thresh, pred):
    X_k, y_k, prediction, x_r, y_r = Remove(X, y, k, scores, dev_index, pred, thresh)

    if y_k.shape[0] == np.sum(y_k) or np.sum(y_k) == 0:  # data contains only one class: 1
        return None

    # Fit the model again
    model_k = LogisticRegression(penalty='l2', C=l2)
    model_k.fit(X_k, y_k)

    # predictthe probaility with test point
    test_point = X["dev"][dev_index]
    test_point = np.reshape(test_point, (1, -1))

    new = model_k.predict_proba(test_point)[0][1]
    return new


# In[4]:



# In[60]:


def recursive_NT(test_idx, old, X_0, y_0, model, l2, thresh, X_dist, I=100, D=1):
    eps = 1 / X_0["train"].shape[0]
    X_l = X_0
    y_l = y_0
    X_r = X_0
    y_r = y_0
    # Compute IP
    F_dev = np.concatenate([X_0["dev"], np.ones((X_0["dev"].shape[0], 1))], axis=1)
    new_hessain = hessian_(X_l, model, l2)
    new_inv = inverse_hessian(new_hessain)
    new_grad_train = gradient_(X_r, y_r, model, l2)
    delta_k = -eps * new_inv @ new_grad_train.T
    grad_f = F_dev[test_idx] * (old * (1 - old))
    delta_pred = grad_f @ delta_k

    K_new = y_0.shape[0]
    predicted_change_new = None
    ite = 0
    diff = K_new - 0
    removed_order = []
    while (ite < I and diff > D and K_new != 1):
        ite += 1

        if old > 0.5:
            sorted_index = np.flip(delta_pred.argsort())
        else:
            sorted_index = delta_pred.argsort()

        for k in range(1, y_r.shape[0]):
            top_k_index = sorted_index[:k]
            predicted_change = -np.sum(delta_pred[top_k_index])


            if ((old < thresh) != (old + predicted_change < thresh)):
                # print("K", k)

                diff = K_new - k
                K_new = k
                predicted_change_new = predicted_change

                index_whole = []
                for idx_r in top_k_index:
                    point = X_r[idx_r]
                    idx_0 = X_dist[str(point.tolist())]
                    index_whole.append(idx_0)

                X_r = X_r[top_k_index]
                y_r = y_r[top_k_index]

                X_l = np.delete(X_0, index_whole, axis=0)
                y_l = np.delete(y_0, index_whole, axis=0)
                # print("removed shape", X_r.shape)
                # print("left shape", X_l.shape)

                # new hessian for the left points
                new_hessain = hessian_(X_l, model, C=l2)
                new_inv = inverse_hessian(new_hessain)
                # new gradient for the removed points
                new_grad_train = gradient_(X_r, y_r, model, C=l2)

                delta_k = -eps * new_inv @ new_grad_train.T
                grad_f = F_dev[test_idx] * (old * (1 - old))
                delta_pred = grad_f @ delta_k
                break

            if k == y_r.shape[0] - 1:
                if K_new == y_0.shape[0] and diff == y_0.shape[0]:
                    return None, predicted_change_new, ite, None, []

                return K_new, predicted_change_new, ite, diff, index_whole

    return K_new, predicted_change_new, ite, diff, index_whole



def IP_iterative(l2, X, y, dataname, thresh):
    model = LogisticRegression(penalty='l2', C=l2)
    model.fit(X["train"], y["train"])
    model.score(X["dev"], y["dev"])

    # compute IP for new train
    from sklearn.preprocessing import normalize
    w = np.concatenate((model.coef_, model.intercept_[None, :]), axis=1)
    F_train = np.concatenate([X["train"], np.ones((X["train"].shape[0], 1))], axis=1) # Concatenating one to calculate the gradient with respect to intercept
    F_dev = np.concatenate([X["dev"], np.ones((X["dev"].shape[0], 1))], axis=1)

    error_train = model.predict_proba(X["train"])[:, 1] - y["train"]
    error_dev = model.predict_proba(X["dev"])[:, 1] - y["dev"]

    gradient_train = F_train * error_train[:, None]
    gradient_dev = F_dev * error_dev[:, None]

    from scipy import sparse
    probs = model.predict_proba(X["train"])[:, 1]
    H = F_train.T @ np.diag(probs * (1 - probs)) @ F_train / X["train"].shape[0] + 1 * np.eye(F_train.shape[1]) / X["train"].shape[0]
    H_inv = np.linalg.inv(H)

    eps = 1 / X["train"].shape[0]
    delta_k = -eps * H_inv @ gradient_train.T
    pred = np.reshape(model.predict_proba(X["dev"])[:, 1], (model.predict_proba(X["dev"])[:, 1].shape[0], 1))
    grad_f = F_dev * (pred * (1 - pred))
    delta_pred = grad_f @ delta_k


    # In[46]:


    X_dist = {}
    for i in range(X["train"].shape[0]):
        X_dist[str(X["train"][i].tolist())] = i

    NT_app_k = []
    new_predictions = []
    iterations = []
    diffs = []
    order_lists = []

    for i in range(X["dev"].shape[0]):
        test_idx = i
        print("test_idx", test_idx)

        old = pred[test_idx].item()
        print("old", old)
        K_nt, pred_change_nt, ite, diff, order= recursive_NT(test_idx, old, X["train"], y["train"], model, l2, thresh, X_dist, I=100, D=1)

        if pred_change_nt != None:
            new_nt = new_train(X, y, l2, K_nt, test_idx, delta_pred, thresh, pred)
        else:
            new_nt = None

        print("K_nt, pred_change_nt, ite, diff")
        print(K_nt, pred_change_nt, ite, diff)
        print("new", new_nt)
        print("order", len(order))
        print()
        NT_app_k.append(K_nt)
        new_predictions.append(new_nt)
        iterations.append(ite)
        diffs.append(diff)
        order_lists.append(order)

    np.save("NT_app_k_"+dataname+ "_LR_I100_D1.npy", NT_app_k)
    np.save("new_predictions_"+dataname+ "_LR_I100_D1.npy", new_predictions)
    np.save("iterations_"+dataname+ "_LR_I100_D1.npy", iterations)
    np.save("diffs_"+dataname+ "_LR_I100_D1.npy", diffs)
    np.save("order_"+dataname+ "_LR_I100_D1.npy", order_lists)


