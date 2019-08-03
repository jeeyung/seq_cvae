import numpy as np

def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=50):
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = X_pred.shape[0]
    # print('pred', X_pred)
    idx_topk_part = np.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    tp = 1. / np.log2(np.arange(2, k + 2))
    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis], idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum() for n in heldout_batch.getnnz(axis=1)])
    return DCG / IDCG

def Recall_at_k_batch(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]
    idx = np.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True
    X_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall

# def Precision_at_k_batch(X_pred, heldout_batch, k=100):
#     batch_users = X_pred.shape[0]
#     idx = np.argpartition(-X_pred, k, axis=1)
#     X_pred_binary = np.zeros_like(X_pred, dtype=bool)
#     X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True
#     X_true_binary = (heldout_batch > 0).toarray()
#     tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(np.float32)
#     precision = tmp / np.minimum(k, X_pred_binary.sum(axis=1))
#     return precision

def AUC_score(X_pred, heldout_batch):
    batch_users = X_pred.shape[0]
    X_true_binary = (heldout_batch > 0).toarray()
    X_false_binary = ~(heldout_batch != 0).toarray()
    x_tr = np.isneginf(X_pred)
    # there is no case where true in X_false_binary and false in X_pred
    X_false_binary_add = np.logical_xor(X_false_binary,x_tr)
    # valid method
    # val = 0
    # for b in range(batch_users):
    #     indicator = 0
    #     i_val = X_pred[b][X_true_binary[b]]
    #     j_val = X_pred[b][X_false_binary_add[b]]
    #     for i in i_val:
    #         for j in j_val:
    #             if i > j:
    #                 indicator += 1
    #     eu = len(i_val) * len(j_val)
    #     val += indicator / eu
    # auc = val / batch_users
    val = 0
    for b in range(batch_users):
        indicator = 0
        i_val = X_pred[b][X_true_binary[b]]
        j_val = X_pred[b][X_false_binary_add[b]]
        i_val = np.expand_dims(i_val, axis=1)
        indicator = (i_val > j_val).sum()
        eu = len(i_val) * len(j_val)
        val += indicator / eu
    auc = val / batch_users
    return auc

def show_recommended_items(X_pred, k=50):
    batch_users = X_pred.shape[0]
    idx_topk_part = np.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    return idx_topk
