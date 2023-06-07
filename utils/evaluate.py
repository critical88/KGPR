from .metrics import *
from .parser import parse_args

from .metrics1 import calc_metrics_at_k
import torch
import numpy as np
import multiprocessing
import heapq
from time import time
from tqdm import tqdm
import pandas as pd
cores = multiprocessing.cpu_count() // 2

args = parse_args()
Ks = eval(args.Ks)
device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
BATCH_SIZE = args.test_batch_size
batch_test_flag = args.batch_test_flag


def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc

def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = AUC(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc

def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(precision_at_k(r, K))
        recall.append(recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(ndcg_at_k(r, K, user_pos_test))
        hit_ratio.append(hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}


def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    # uid
    u = x[1]
    # user u's items in the training set
    try:
        training_items = train_user_set[u]
    except Exception:
        training_items = []
    # user u's items in the test set
    user_pos_test = test_user_set[u]

    all_items = set(range(0, n_items))

    test_items = list(all_items - set(training_items))
    rating[training_items] = -np.inf
    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    performance = get_performance(user_pos_test, r, auc, Ks)

    # rating = rating.argsort()[::-1][0:100]
    # rating[rating.argsort()[::-1]] = range(len(rating))
    # rating = rating.astype(int)
    # performance["rating"] = rating
    return performance
def evaluate(model, test_batch_size, n_items, train_user_dict, test_user_dict, device):

    model.eval()

    user_ids = list(test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]

    item_ids = torch.arange(n_items, dtype=torch.long).to(device)

    cf_scores = []
    metric_names = ['precision', 'recall', 'ndcg']
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}

    embeddings = model.generate()
    new_user_embedding, new_item_embedding = embeddings[0], embeddings[1]
    with tqdm(total=len(user_ids_batches), desc='Evaluating Iteration') as pbar:
        for batch_user_ids in user_ids_batches:
            batch_user_ids = batch_user_ids.to(device)

            with torch.no_grad():
                
                user_embedded = new_user_embedding[batch_user_ids]
                item_embedded = new_item_embedding[item_ids]
                batch_scores = torch.matmul(user_embedded, item_embedded.transpose(0, 1))
                # batch_scores = torch.mul(user_embedded, item_embedded).sum(dim=-1, keepdim=False)

            batch_scores = batch_scores.cpu()
            batch_metrics = calc_metrics_at_k(batch_scores, train_user_dict, test_user_dict, batch_user_ids.cpu().numpy(), item_ids.cpu().numpy(), Ks)

            cf_scores.append(batch_scores.numpy())
            for k in Ks:
                for m in metric_names:
                    metrics_dict[k][m].append(batch_metrics[k][m])
            pbar.update(1)

    # cf_scores = np.concatenate(cf_scores, axis=0)
    for k in Ks:
        for m in metric_names:
            metrics_dict[k][m] = np.concatenate(metrics_dict[k][m]).mean()
    return metrics_dict

def get_orginal_kg(model):
    """
    
    """
    n_relation = (model.n_relations - 1) / 2 - 1
    head, tail = model.edge_index
    edge_type = model.edge_type - 1

    org_index = torch.where(edge_type <= n_relation)[0]

    can_triplets_np = np.array([head[org_index].cpu().numpy(), edge_type[org_index].cpu().numpy(), tail[org_index].cpu().numpy()])

    can_triplets_np = can_triplets_np.T

    return can_triplets_np

def get_masked_info(model, mask):
    """
    mask: (N_triplets * 2)

    as we convert the directed graph to undirected graph in the training stage, 
    the dimension of mask need to be doubled 

    however, the augmented edges are useless in pruning stage, thus we pick them out in this method.

    return masked_triplet_df (DataFrame), all_triplets_df (DataFrame)

    all of them without the virtual edges.
    """
    n_relation = (model.n_relations - 1) / 2 - 1
    
    head, tail = model.edge_index
    edge_type = model.edge_type - 1

    can_triplets_np = get_orginal_kg(model)

    head, tail, edge_type = head.cpu().numpy(), tail.cpu().numpy(), edge_type.cpu().numpy()
    

    removed_triplets = np.array([head[mask], edge_type[mask], tail[mask]]).transpose(1, 0)
    
    
    masked_triplest_df = pd.DataFrame(removed_triplets, columns=["h", "r", "t"])
    
    all_triplets_df = pd.DataFrame(can_triplets_np, columns=["h", "r", "t"])

    masked_triplest_df["match"] = masked_triplest_df["h"].astype(str).str.cat(masked_triplest_df["r"].astype(str), sep='_').str.cat(masked_triplest_df["t"].astype(str), sep="_")
    
    all_triplets_df["match"] = all_triplets_df["h"].astype(str).str.cat(all_triplets_df["r"].astype(str), sep='_').str.cat(all_triplets_df["t"].astype(str), sep="_")

    return masked_triplest_df, all_triplets_df

def save_unpruned_node(model, mask_file, saved_pruned_kg_file):
    """
    mask_file: str, the location of mask_info 
    saved_pruned_kg_file: str, the location to be saved in.
    as we save the mask info in the update_q_mask in modules/MaskePrundNode.py,
    this method is designed for directly read the saved mask_file to get the pruned_kg
    """
    mask = np.load(mask_file)
    masked_triplest_df, all_triplets_df = get_masked_info(model, mask)
    
    intersected_triplets = all_triplets_df[~(all_triplets_df["match"].isin(masked_triplest_df["match"]))]
    intersected_triplets[["h", "r", "t"]].to_csv(saved_pruned_kg_file, index=False, header=None, sep=" ")

def test(model, user_dict, n_params, epoch=-1):
    result = {'precision': np.zeros(len(Ks)),
              'recall': np.zeros(len(Ks)),
              'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)),
              'auc': 0., 
              "ratings": []}

    global n_users, n_items
    n_items = n_params['n_items']
    n_users = n_params['n_users']

    global train_user_set, test_user_set
    train_user_set = user_dict['train_user_set']
    test_user_set = user_dict['test_user_set']

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE
    i_batch_size = BATCH_SIZE

    test_users = list(test_user_set.keys())
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    embs = model.generate(epoch)

    if len(embs) == 2:
        entity_gcn_emb, user_gcn_emb = embs
    else:
        entity_gcn_emb, user_gcn_emb, nomask = embs
        masked_triplest_df, all_triplets_df = get_masked_info(model, model.final_masked_edges)
        intersected_triplets = all_triplets_df[(all_triplets_df["match"].isin(masked_triplest_df["match"]))]
        print("current sparsity:%.4f, total sparsity:%.4f" % (len(nomask[nomask < model.alpha]) / len(nomask), len(intersected_triplets) / len(all_triplets_df)))

    for u_batch_id in tqdm(range(n_user_batchs)):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_list_batch = test_users[start: end]
        user_batch = torch.LongTensor(np.array(user_list_batch)).to(device)
        u_g_embeddings = user_gcn_emb[user_batch]

        if batch_test_flag:
            # batch-item test
            n_item_batchs = n_items // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), n_items))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, n_items)

                item_batch = torch.LongTensor(np.array(range(i_start, i_end))).view(i_end-i_start).to(device)
                i_g_embddings = entity_gcn_emb[item_batch]

                i_rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach().cpu()

                rate_batch[:, i_start: i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == n_items
        else:
            # all-item test
            item_batch = torch.LongTensor(np.array(range(0, n_items))).view(n_items, -1).to(device)
            i_g_embddings = entity_gcn_emb[item_batch]
            rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach().cpu()

        user_batch_rating_uid = zip(rate_batch, user_list_batch)
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users
            result['auc'] += re['auc']/n_test_users
            # result["ratings"].append(re["rating"])
    assert count == n_test_users
    pool.close()
    return result
