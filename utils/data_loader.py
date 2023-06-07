import numpy as np
from tqdm import tqdm
import networkx as nx
import scipy.sparse as sp

import random
from time import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

n_users = 0
n_items = 0
n_entities = 0
n_relations = 0
n_nodes = 0
train_user_set = defaultdict(list)
test_user_set = defaultdict(list)
train_item_set = defaultdict(list)

def read_cf(file_name):
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]

        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])

    return np.array(inter_mat)


def remap_item(train_data, test_data):
    global n_users, n_items
    n_users = max(max(train_data[:, 0]), max(test_data[:, 0])) + 1
    n_items = max(max(train_data[:, 1]), max(test_data[:, 1])) + 1

    for u_id, i_id in train_data:
        train_user_set[int(u_id)].append(int(i_id))
        train_item_set[int(i_id)].append(int(u_id))
    for u_id, i_id in test_data:
        test_user_set[int(u_id)].append(int(i_id))


def read_triplets(file_name):
    global n_entities, n_relations, n_nodes

    can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
    can_triplets_np = np.unique(can_triplets_np, axis=0)

    if args.inverse_r:
        # get triplets with inverse direction like <entity, is-aspect-of, item>
        inv_triplets_np = can_triplets_np.copy()
        inv_triplets_np[:, 0] = can_triplets_np[:, 2]
        inv_triplets_np[:, 2] = can_triplets_np[:, 0]
        inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1
        # consider two additional relations --- 'interact' and 'be interacted'
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        inv_triplets_np[:, 1] = inv_triplets_np[:, 1] + 1
        # get full version of knowledge graph
        triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)
    else:
        # consider two additional relations --- 'interact'.
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        triplets = can_triplets_np.copy()
    
    n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1  # including items + users
    n_nodes = n_entities + n_users
    n_relations = max(triplets[:, 1]) + 1

    return triplets


def build_graph(train_data, triplets):
    ckg_graph = nx.MultiDiGraph()
    rd = defaultdict(list)

    print("Begin to load interaction triples ...")
    for u_id, i_id in tqdm(train_data, ascii=True):
        rd[0].append([u_id, i_id])

    print("\nBegin to load knowledge graph triples ...")
    for h_id, r_id, t_id in tqdm(triplets, ascii=True):
        ckg_graph.add_edge(h_id, t_id, key=r_id)
        rd[r_id].append([h_id, t_id])

    return ckg_graph, rd


def build_sparse_relational_graph(relation_dict, graph):
    def _bi_norm_lap(adj):
        # D^{-1/2}AD^{-1/2}
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    adj_mat_list = []
    horz_idx = np.array([], dtype=int)
    vert_idx = np.array([], dtype=int)
    print("Begin to build sparse relation matrix ...")
    for r_id in tqdm(relation_dict.keys()):
        np_mat = np.array(relation_dict[r_id])
        if r_id == 0:
            cf = np_mat.copy()
            cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items)
            vals = [1.] * (len(cf))
            
            # horz_idx = np.concatenate((horz_idx, cf[:, 0]))
            # vert_idx = np.concatenate((vert_idx, cf[:, 1]))
            adj = sp.coo_matrix((vals, (cf[:, 0], cf[:, 1])), shape=(n_nodes, n_nodes))
        else:
            np_mat1 = np.zeros((np_mat.shape[0], 3), dtype=int)
            np_mat1[:, :2] = np_mat
            np_mat1[:, 2] = r_id
            vals = [1.] * len(np_mat)
            # horz_idx = np.concatenate((horz_idx, np_mat[:, 0] + n_users))
            # vert_idx = np.concatenate((vert_idx, np_mat[:, 1] + n_users))
            adj = sp.coo_matrix((vals, (np_mat[:, 0] + n_users, np_mat[:, 1] + n_users)), shape=(n_nodes, n_nodes))
            pass
        adj_mat_list.append(adj)
    
    # vals = [1.] * len(horz_idx)
    # adj = sp.coo_matrix((vals, (horz_idx, vert_idx)), shape=(n_nodes, n_nodes))
    # sec_order_adj = adj.dot(adj)[:n_users, :].tocoo()
    # 每个entity涉及的用户数
    # _psum = np.sum(sec_order_adj > 0, axis=0)
    # 每个用户涉及的总次数
    # _usum = np.sum(sec_order_adj, axis=1)
    # tf-idf
    # tf，该entity在user中出现的次数 / 该user总的entity的出现次数

    # tf = _si_norm_lap(sec_order_adj)
    # idf， 总user数 / 包含改entity的用户数
    
    # idf = -1 * np.log2((_psum +1 )/ n_users)
    # d_inv = np.power(_psum, -1).flatten()
    # d_inv[np.isinf(d_inv)] = 0.
    # d_mat_inv = sp.diags(d_inv)

    # norm_adj = d_mat_inv.dot(adj)
    # return norm_adj.tocoo()
    
    # tf_idf = tf * idf
    norm_mat_list = [_bi_norm_lap(mat) for mat in adj_mat_list]
    mean_mat_list = [_si_norm_lap(mat) for mat in adj_mat_list]
    # interaction: user->item, [n_users, n_entities]
    norm_mat_list[0] = norm_mat_list[0].tocsr()[:n_users, n_users:].tocoo()
    mean_mat_list[0] = mean_mat_list[0].tocsr()[:n_users, n_users:].tocoo()
    # deg = np.zeros((n_nodes, 1))
    # for i, mat in enumerate(adj_mat_list):
    #     if i == 0:
    #         deg += np.array(mat.sum(1))
    #     else:
    #         deg[n_users+n_items:] += np.array(mat.sum(1)[n_users+n_items:])



    return adj_mat_list, norm_mat_list, mean_mat_list



def load_data(model_args):
    global args
    args = model_args
    directory = args.data_path + args.dataset + '/'

    print('reading train and test user-item set ...')
    train_cf = read_cf(directory + 'train.txt')
    test_cf = read_cf(directory + 'test.txt')
    remap_item(train_cf, test_cf)

    print('combinating train_cf and kg data ...')
    triplets = read_triplets(directory + args.kg_file + '.txt')

    print('building the graph ...')
    graph, relation_dict = build_graph(train_cf, triplets)

    print('building the adj mat ...')
    adj_mat_list, norm_mat_list, mean_mat_list = build_sparse_relational_graph(relation_dict, graph)

    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
        'n_entities': int(n_entities),
        'n_nodes': int(n_nodes),
        'n_relations': int(n_relations)
    }
    user_dict = {
        'train_user_set': train_user_set,
        'test_user_set': test_user_set,
        'train_item_set': train_item_set
    }
    return train_cf, test_cf, user_dict, n_params, graph, \
           [adj_mat_list, norm_mat_list, mean_mat_list]

