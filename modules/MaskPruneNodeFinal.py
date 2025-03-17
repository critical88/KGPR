'''
Created on July 1, 2020
PyTorch Implementation of KGIN
@author: Tinglin Huang (tinglin.huang@zju.edu.cn)
'''
__author__ = "huangtinglin"

import random
import numpy as np
import math
import torch
import os
from tqdm import tqdm
from time import time
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add, scatter_sum, scatter_max, scatter_min
from torch_scatter.utils import broadcast
from torch_scatter.composite import scatter_softmax
from collections import defaultdict
from typing import List

def torch_isin(a, b, device=None):
        _, a_counts = a.unique(return_counts=True)
        a_cat_b, combined_counts = torch.cat([a, b]).unique(return_counts=True)
        return (combined_counts - a_counts).gt(0)[a]
def torch_np_isin(a, b, device=None):
        a = a.detach().cpu().numpy()
        b = b.detach().cpu().numpy()
        return torch.BoolTensor(np.isin(a, b)).to(device)
class Aggregator(nn.Module):
    """
    Relational Path-aware Convolution Network
    """
    def __init__(self, n_users):
        super(Aggregator, self).__init__()
        self.linear = nn.Linear(64, 64)
        
        self.gate1 = nn.Linear(64, 64, bias=False)
        self.gate2 = nn.Linear(64, 64, bias=False)
        self.n_users = n_users

    def forward(self, entity_emb, user_emb, entity_2nd_emb, ent_weight_emb,
                edge_index, edge_type, interact_mat, weight, unmask):

        n_entities = entity_emb.shape[0]
        channel = entity_emb.shape[1]
        n_users = self.n_users

        """KG aggregate"""
        ## all triples
        head, tail = edge_index
        edge_relation_emb = weight[edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        # corresponding tail entity * relation 
        neigh_relation_emb = (entity_emb)[tail] * (unmask.unsqueeze(-1))  * edge_relation_emb   # + entity_emb[head] # [-1, channel]
        
        ##  GCN
        entity_agg = scatter_sum(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)

        """user aggregate"""
        user_agg = torch.sparse.mm(interact_mat, entity_emb)

        return entity_agg, user_agg




class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, channel, n_hops, n_users,n_entities, n_relations, interact_mat, 
                 ind,  node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.interact_mat = interact_mat
        self.n_relations = n_relations
        self.n_users = n_users
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind
        self.ent_gru = nn.GRU(channel, channel)

        self.ent_linear = nn.Linear(128, 64)
        self.user_linear = nn.Linear(128, 64)
        initializer = nn.init.xavier_uniform_
        

        for i in range(n_hops):
            self.convs.append(Aggregator(n_users=n_users))

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout


    

    def forward(self, user_emb, entity_emb, entity_2nd_emb, user_2nd_emb, edge_index, edge_type,
                interact_mat, weight, triplet_mask,  mess_dropout=True, node_dropout=False, training_epoch=-1):

        entity_res_emb = entity_emb  # [n_entity, channel]
        entity_2nd_res_emb = entity_2nd_emb
        user_res_emb = user_emb   # [n_users, channel]
        user_2nd_res_emb = user_emb
        entity_layer_embed = []
        user_layer_embed = []
        # head, tail = edge_index
        # edge_relation_emb = self.weight[edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        # neigh_relation_emb = entity_emb[tail] 
        # edge_norm = (entity_emb[head] + edge_relation_emb - entity_emb[tail]).norm(dim=1)
        # y_soft = edge_norm
        # threshold = 0.1
        # # 大于该阈值就可以保留下来
        # index = torch.where(y_soft >= threshold, 1, 0)
        # y_hard = torch.zeros_like(y_soft, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
        # mask = y_hard - y_soft.detach() + y_soft
        # unmask = 1 - mask
        
        # if training_epoch >= 0:

        #     # # gumbel STE
        #     y_soft = F.gumbel_softmax(triplet_mask)
        #     # triplets = y_soft.detach().cpu().numpy()
        #     # threshold = np.percentile(triplets[:, 0], np.tanh(training_epoch / 100) * 20)

        #     # # gumbel hard
        #     # y_soft = F.gumbel_softmax(triplet_mask, hard=True)
        #     # # mask 0列代表不需要mask，
        #     # unmask =  mask[:, 0] * (1 - q_mask)

        #     ## soft mask
        #     # mask = 1 - y_soft[:, 1]
        #     ## threshold hard mask 
        #     # threshold = 0.2
        #     # y_soft = y_soft[:, 1]
        #     # # 大于该阈值就可以保留下来
        #     # index = torch.where(y_soft >= threshold, 1, 0)
        #     # y_hard = torch.zeros_like(y_soft, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
        #     # mask = y_hard - y_soft.detach() + y_soft
        #     # uniform
        #     # threshold = 0.2
        #     # y_hard = torch.where(triplet_mask < threshold, 1, 0)
        #     # mask = y_hard - triplet_mask.detach() + triplet_mask
        #     mask = 1 - F.tanh(triplet_mask * 5)
        #     mask = torch.clamp(mask, 0, 1)
        #     unmask = mask * (1-q_mask)
        # else:
        #     # mask 1列代表需要mask，0列大于1列表示保留的信息
        #     # mask = F.softmax(triplet_mask, dim=1)
        #     unmask = (1 - q_mask)
        unmask = triplet_mask
        for i in range(len(self.convs)):
            entity_emb, user_emb = self.convs[i](entity_emb, user_emb, entity_2nd_emb, user_2nd_emb,
                                                 edge_index, edge_type, interact_mat, weight, unmask)

            """message dropout"""
            if mess_dropout:
                # entity_2nd_emb = self.dropout(entity_2nd_emb)
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
                # user_2nd_emb = self.dropout(user_2nd_emb)
            
            entity_layer_embed.append(entity_emb)
            user_layer_embed.append(user_emb)

            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)
            
            # weight = (len(self.convs) - i) / len(self.convs)
            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb )
            user_res_emb = torch.add(user_res_emb, user_emb)

            # entity_2nd_res_emb = torch.add(entity_2nd_res_emb, entity_2nd_emb)
            # user_2nd_res_emb = torch.add(user_2nd_res_emb, user_2nd_emb)
        if training_epoch >= 0:
            return entity_res_emb, user_res_emb
        else:
            return entity_res_emb, user_res_emb, unmask
        # return torch.cat([entity_res_emb, entity_2nd_res_emb], dim=1), torch.cat([user_res_emb, user_2nd_res_emb], dim=1)


class ScoreEstimator(nn.Module):

    def __init__(self, dim):
        super(ScoreEstimator, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim, bias=False)
        self.layer2 = nn.Linear(self.dim, 1, bias=False)
        # self.layer3 = nn.Linear(self.dim * 2, self.dim)
        # self.layer4 = nn.Linear(self.dim, 1)
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        # nn.init.xavier_uniform_(self.layer3.weight)
        # nn.init.xavier_uniform_(self.layer4.weight)

    def forward(self, ent_embeddings, head, tail, dim_size):
        
        x = F.relu(self.layer1(ent_embeddings))
        
        ent_scores = F.sigmoid(self.layer2(x))

        return ent_scores.squeeze(1)
        # edge = torch.cat([ent_embeddings[head], ent_embeddings[tail]], dim=-1)
        # x = F.relu(self.layer3(edge))
        # x = F.sigmoid(self.layer4(x))
        # gate = scatter_softmax(src=x, index=tail, dim=0)

        # x = F.relu(self.layer1(ent_embeddings))
        # ent_score = F.sigmoid(self.layer2(x))
        # agg_ent_score = scatter_mean(src=ent_score[head] , index=tail, dim=0, dim_size=dim_size)

        # ent_score = ent_score.squeeze(1)
        # # ent_score = gate * ent_score
        # agg_ent_score = agg_ent_score.squeeze(1)

        # return torch.min(torch.stack([ent_score, agg_ent_score]), dim=0).values
        # x = F.relu(self.layer1(ent_embeddings))
        # ent_score = F.sigmoid(self.layer2(x))
        # scores = ent_score[head]
        # ent_scores = scatter_mean(scores, index=tail, dim=0, dim_size=dim_size)
        # return ent_scores.squeeze(1)

class Recommender(nn.Module):
    def __init__(self, data_config, args_config, graph, adj_mat):
        super(Recommender, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities
        self.epoch = args_config.epoch
        self.sparsity = args_config.sparsity
        self.c = -math.log(1-self.sparsity) / self.epoch
        self.decay = args_config.l2
        self.save_dir = args_config.save_dir

        self.args_config = args_config
        self.sim_decay = args_config.sim_regularity
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.ind = args_config.ind
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
                                                                      else torch.device("cpu")
        self.remove_ratio = args_config.remove_ratio
        self.alpha = args_config.alpha

        self.beta = args_config.beta
        
        self.gamma = args_config.gamma

        self.sample_num = args_config.sample_num

        self.tau = torch.FloatTensor([0.2])

        self.tau = nn.Parameter(self.tau)

        self.margin = 0.7
        self.temperature = 0.2
        self.adj_mat = adj_mat
        self.graph = graph
        self.edge_index, self.edge_type,  = self._get_edges(graph)
        self.need_sampled_entity_idx = None

        self.edge_cnt = self.edge_index.shape[1]
        
        self.triplet_mask = torch.zeros(self.n_entities) 

        # if self.args_config.method == "g":
        #     self.triplet_mask += 0.5
        
        # self.triplet_mask = torch.zeros(self.edge_index.shape[1]) 

        self.triplet_mask = nn.Parameter(self.triplet_mask)


        # self.layer1 = nn.Linear(self.emb_size, self.emb_size)
        # self.layer2 = nn.Linear(self.emb_size, 1)
        # self.layer3 = nn.Linear(3, self.emb_size)
        # self.layer4 = nn.Linear(self.emb_size, 1)
        # nn.init.xavier_uniform_(self.layer1.weight)
        # nn.init.xavier_uniform_(self.layer2.weight)
        # nn.init.xavier_uniform_(self.layer3.weight)
        # nn.init.xavier_uniform_(self.layer4.weight)

        ### tail to edge_index map, in order to mapping the user specific mask
        self.final_masked_edges = torch.LongTensor(range(self.edge_index.shape[1]))
        self.saved_ent_scores = np.empty((0, self.n_entities))
        self.q_mask = torch.zeros(self.edge_cnt).to(self.device)
        # # self.q_mask = nn.Parameter(self.q_mask)
        self.q_mask.requires_grad = False
        # prioritize the no mask
        # self.triplet_mask[:, 1] += 1
        
        self._init_weight()
        i = self.interact_mat._indices()
        v = torch.FloatTensor([1.0] * i.shape[1]).to(self.device)
        self.item_interact_mat = torch.sparse.FloatTensor(i, v, torch.Size((self.n_users, self.n_items))).to(self.device)
        self._init_user2ent()

        self.all_embed = nn.Parameter(self.all_embed)
        self.weight = nn.Parameter(self.weight)  
        self.gcn = self._init_model()
        self.score_estimator = ScoreEstimator(self.emb_size)

    def _init_user2ent(self):
        ## create user-edge sparse matrix
        # self.user2ent_mat = torch.sparse.mm(self.item_interact_mat, self.item2ent_graph)
        if not os.path.exists("user2ent"):
            os.makedirs("user2ent")
        user_edge_path = os.path.join("user2ent", f"{self.args_config.dataset}_{self.args_config.is_two_hop}_{self.args_config.num_sample_user2ent}.pt")
        
        sample_user2ent_num = self.args_config.num_sample_user2ent
        if not os.path.exists(user_edge_path):
            all_head, all_tail = self.edge_index.detach().cpu().numpy()
            ent2user = [np.array([], dtype=int) for _ in range(self.n_entities)]
            interact_idx = self.item_interact_mat._indices().detach().cpu().numpy()
            for t in interact_idx.T: 
                # ent2user[t[1]].add(t[0])
                ent2user[t[1]] = np.append(ent2user[t[1]], np.array(t[0], dtype=int))
            head = np.arange(self.n_items)
            n_users = self.n_users
            def propagate(head, n_users, ent2user, depth=0):
                if depth==3:
                    return ent2user
                threshold = n_users // 3
                # edge_idxs = np.where((np.isin(all_head, head)))[0]
                uq_tail = np.unique(all_tail[np.isin(all_head, head)])
                next_head = []
                for t in tqdm(uq_tail, total=len(uq_tail)):
                    if len(ent2user[t]) > 0:
                        continue
                    next_head.append(t)
                    ent2user[t] = np.unique(np.concatenate([ent2user[h] for h in np.unique(all_head[all_tail == t])]))

                
                # for e in tqdm(edge_idxs, total=len(edge_idxs)):
                #     h, t = all_head[e], all_tail[e]
                #     if len(ent2user[h]) >= threshold:
                #         continue
                #     if len(ent2user[t]) > 0:
                #         continue
                #     next_head.append(t)
                    
                #     ent2user[t] = ent2user[t].union(ent2user[h])
                next_head = np.unique(np.array(next_head))
                return propagate(next_head, n_users, ent2user, depth= depth+1)
            ent2user = propagate(head, n_users, ent2user)
            row, col = [], []
            for i, users in tqdm(enumerate(ent2user), total=len(ent2user)):
                if len(users) <= sample_user2ent_num:
                    row += list(users)
                    col += [i] * len(users)
                else:
                    
                    row += list(np.random.choice(np.array(list(users)), size=sample_user2ent_num, replace=False))
                    col += [i] * sample_user2ent_num
            
            row = torch.LongTensor(row)
            col = torch.LongTensor(col)
            idx = torch.stack([row, col], dim=0)
            torch.save(idx, user_edge_path)
            # self.item2ent_graph = self.build_item2entities_graph(self.edge_index)
            # user2ent = torch.sparse.mm(self.item_interact_mat, self.item2ent_graph)
            # user_idx = user2ent._indices()
            # all_tail = user_idx[1]
            # bincount = all_tail.bincount(minlength=self.n_entities)
            
            # preserved_entities = torch.where(bincount <= sample_user2ent_num)[0]
            # wait4sampled_entities = torch.where(bincount > sample_user2ent_num)[0]
            # ## reassign the values to 1
            # start_time = time()
            # print("start sample user2ent tail edge")
            # t_2hop_sample_edge_idx = []
            # # rand_order_samples = torch.randint(0, self.n_users, (len(self.t_2hop_tail_entities_bin_sample),sample_user2ent_num * 3)).to(self.device)
            # for entity_id in tqdm(wait4sampled_entities, total=wait4sampled_entities.shape[0]):
            #     t_2hop_edge_sample_ord = torch.where(all_tail == entity_id)[0] #tail2edge_matrix[entity_id]._indices()[0] #
            #     ent_size = t_2hop_edge_sample_ord.shape[0]
            #     # sample_num = sample_user2ent_num
            #     # t_2hop_edge_sample_ord = t_2hop_edge_sample_ord[sampled_idx]
            #     if ent_size > 10000:
            #         sampled_idx = set()
            #         while len(sampled_idx) < sample_user2ent_num:
            #             sampled_idx.add(random.randint(0, ent_size-1))
            #         t_2hop_edge_sample_ord = t_2hop_edge_sample_ord[list(sampled_idx)]
            #     else:
            #     # torch choice
            #     # t_2hop_edge_sample_ord = torch_isin(t_2hop_edge_sample_ord, all_tail)
            #         t_2hop_edge_sample_ord = t_2hop_edge_sample_ord[torch.randperm(t_2hop_edge_sample_ord.shape[0])[:sample_user2ent_num]]
            #     t_2hop_sample_edge_idx.append(t_2hop_edge_sample_ord)
            # end_time = time()
            # print("sample the user2end tail edge, consume: %.2fs" % (end_time - start_time))
            # if len(t_2hop_sample_edge_idx) > 0:
            #     t_2hop_sample_edge_idx = torch.cat(t_2hop_sample_edge_idx)
            # else:
            #     t_2hop_sample_edge_idx = torch.LongTensor([]).to(self.device)
            # preserved_edges = torch.where(torch_np_isin(all_tail, preserved_entities, self.device))[0]
            # edge_idx = torch.cat([t_2hop_sample_edge_idx, preserved_edges]).unique()
            # row, col = user_idx[0][edge_idx], user_idx[1][edge_idx]
            

            # idx = torch.stack([row, col], dim=0)
            
            # torch.save(idx, user_edge_path)
        else:
            idx = torch.load(user_edge_path)
        idx = idx.to(self.device)
        idx = torch.cat([idx, self.interact_mat._indices()], dim=-1)
        v = torch.FloatTensor([1.0] * idx.shape[1]).to(self.device)
        self.user2ent_mat = torch.sparse.FloatTensor(idx, v, torch.Size([self.n_users, self.n_entities])).to(self.device)

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))
        self.user_2nd_emb = initializer(torch.empty(self.n_users, self.emb_size))
        self.ent_weight_emb = initializer(torch.empty(self.n_entities, self.emb_size))
        self.entity_2nd_emb = initializer(torch.empty(self.n_entities, self.emb_size))
        self.weight = initializer(torch.empty(self.n_relations - 1, self.emb_size))  # not include interact # [n_relations - 1, in_channel]
        # [n_users, n_entities]
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        return GraphConv(channel=self.emb_size,
                         n_hops=self.context_hops,
                         n_users=self.n_users,
                         n_entities=self.n_entities,
                         n_relations=self.n_relations,
                         interact_mat=self.interact_mat,
                         ind=self.ind,
                         node_dropout_rate=self.node_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate,
                         )

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]

        
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)
    

    # def build_item2entities_graph(self, edge_index):
    #     # 1st order
    #     all_head, all_tail = edge_index
    #     sample_1hop_entities = 100
    #     sample_2_hop_entities = 20

    #     # cache
    #     if self.need_sampled_entity_idx is None:
    #         # item_id is lower than n_items
    #         # 1hop 从items出发，对于一些流行的entities，即大于100个items与其相连的，则只随机取100个item存入item2graph中
    #         # 随后从这些1hop-entities 出发，找2跳的entity，注意我们只找那些未曾见过的entity，即纯粹的2-hop，不往回走
    #         t_1hop_tail_edges = all_tail[all_head < self.n_items]

            
    #         bincount_1hop_tail = t_1hop_tail_edges.bincount(minlength=self.n_entities)
    #         preserved_entities = torch.where((bincount_1hop_tail <= sample_1hop_entities) & (bincount_1hop_tail > 0))[0]
    #         wait4sampled_entities = torch.where(bincount_1hop_tail > sample_1hop_entities)[0]

    #         edge_1hop_condition = torch.where((all_head < self.n_items) & (torch_np_isin(all_tail, preserved_entities, self.device)))[0]
    #         # edge_1hop_condition = torch.where(all_head < self.n_items)[0]
    #         for entity_id in wait4sampled_entities:
    #             t_1hop_edge_sample_ord = torch.where((all_tail == entity_id) & (all_head < self.n_items) )[0]
    #             sample_idx = torch.randint(t_1hop_edge_sample_ord.shape[0], (sample_1hop_entities,))
    #             t_1hop_edge_sample_ord = t_1hop_edge_sample_ord[sample_idx]
    #             edge_1hop_condition = torch.cat((edge_1hop_condition, t_1hop_edge_sample_ord))
    #         # item
    #         self.row_1st = all_head[edge_1hop_condition]
    #         # edge
    #         self.col_1st = all_tail[edge_1hop_condition]


    #         # item2edge_graph = torch.sparse.FloatTensor(torch.stack([items, col_1st], dim=0), torch.FloatTensor(vals), torch.Size((self.n_items, len(graph_tensor)))).to(self.device)
            
    #         # 2nd order 
    #         # in order to get 2nd order martix, 
    #         # we first prepared a full relation item-entity matrix, whose value equals 1 if there exists relation between item and entity
    #         # then we ought to get the 2nd order entity-edge martix, the entity is the tail of 1st order link
            
    #         # build full relation item-entity matrix
            
    #         vals = [1] * len(self.col_1st)
    #         self.full_relation_i2e_mat = torch.sparse.FloatTensor(torch.stack([self.row_1st, self.col_1st], dim=0), torch.FloatTensor(vals).to(self.device), torch.Size((self.n_items, self.n_entities))).to(self.device)
    #         # build entity-edge matrix
    #         # the edge must statisfy two criteria:
    #         # 1. the head of the edge must be the tail of the 1st order edge
    #         # 2. the head of the edge must have less than 20 edges
    #         # first we need to find the start_point, i.e. the tail of the 1st order
    #         t_1hop_tail_entities_uq = t_1hop_tail_edges.unique()
    #         # then we get the head node of all edge
            
    #         # if head node of the edge is in the start point, then this edge will be filled in the entity-edge matrix 
    #         self.edge_2hop_ord_all = torch.where(torch_np_isin(all_head, t_1hop_tail_entities_uq, self.device))[0]
            
    #         #### Attention ####
    #         #### we omit the overwhelming entity-edge, i.e. the entity has more than 10 edges
    #         t_2hop_tail_entities = all_tail[self.edge_2hop_ord_all]
    #         bincount = t_2hop_tail_entities.bincount(minlength=self.n_entities)
    #         seen_1hop_entities = torch.cat([torch.arange(self.n_items).to(self.device), self.row_1st, self.col_1st]).unique()
    #         unseen_1hop_entities = np.setdiff1d(np.arange(self.n_entities), seen_1hop_entities.cpu().numpy())
    #         unseen_1hop_entities = torch.LongTensor(unseen_1hop_entities).to(self.device)
    #         self.t_2hop_tail_entities_bin_allin = torch.where((bincount <= sample_2_hop_entities) & (bincount > 0))[0]
    #         self.t_2hop_tail_entities_bin_allin = np.intersect1d(unseen_1hop_entities.cpu().numpy(), self.t_2hop_tail_entities_bin_allin.cpu().numpy())
    #         self.t_2hop_tail_entities_bin_allin = torch.LongTensor(self.t_2hop_tail_entities_bin_allin).to(self.device)
    #         self.t_2hop_tail_entities_bin_sample = torch.where(bincount > sample_2_hop_entities)[0]
    #         self.t_2hop_tail_entities_bin_sample = np.intersect1d(unseen_1hop_entities.cpu().numpy(), self.t_2hop_tail_entities_bin_sample.cpu().numpy())
    #         self.t_2hop_tail_entities_bin_sample = torch.LongTensor(self.t_2hop_tail_entities_bin_sample).to(self.device)

        
    #     t_2hop_sample_edge_idx = []
    #     start_time = time()
    #     if self.args_config.is_two_hop:
    #         _row = all_tail
    #         # col is the edge indices
    #         _col = torch.arange(all_tail.shape[0]).to(self.device)
    #         vals = [1] * len(_row)
    #         tail2edge_matrix = torch.sparse.FloatTensor(torch.stack([_row, _col], dim=0), torch.FloatTensor(vals).to(self.device), torch.Size((self.n_entities, all_tail.shape[0]))).to(self.device)
    #         print("start sample 2hop tail edge")
    #         # rand_order_samples = torch.randint(0, self.n_entities, (len(self.t_2hop_tail_entities_bin_sample),sample_2_hop_entities * 3)).to(self.device)
    #         for entity_id in tqdm(self.t_2hop_tail_entities_bin_sample, total=self.t_2hop_tail_entities_bin_sample.shape[0]):
    #             t_2hop_edge_sample_ord = torch.where(all_tail == entity_id)[0]
    #             # numpy choice
    #             # t_2hop_edge_sample_ord = np.intersect1d(t_2hop_edge_sample_ord.cpu().numpy(), self.edge_2hop_ord_all)
    #             # t_2hop_edge_sample_ord = np.random.choice(t_2hop_edge_sample_ord, sample_2_hop_entities, replace=False)
    #             # torch choice
    #             sampled_edge_idx = torch_np_isin(t_2hop_edge_sample_ord, self.edge_2hop_ord_all, self.device)
    #             t_2hop_edge_sample_ord = t_2hop_edge_sample_ord[sampled_edge_idx]
    #             t_2hop_edge_sample_ord = t_2hop_edge_sample_ord[torch.randperm(t_2hop_edge_sample_ord.shape[0])[:sample_2_hop_entities]]
    #             t_2hop_sample_edge_idx.append(t_2hop_edge_sample_ord)
    #     end_time = time()
    #     print("sample the 2hop tail edge, consume: %.2fs" % (end_time - start_time))
    #     if len(t_2hop_sample_edge_idx) > 0:
    #         t_2hop_sample_edge_idx = torch.cat(t_2hop_sample_edge_idx)
    #     else:
    #         t_2hop_sample_edge_idx = torch.LongTensor([]).to(self.device)
    #     ## edge_idx exclude the nodes that link more than 10 edges
    #     t_2hop_allin_edge_idx = np.where(np.isin(all_tail.detach().cpu().numpy(), self.t_2hop_tail_entities_bin_allin.detach().cpu().numpy()))[0]
    #     # the nodes that has more than 10 edges
    #     t_2hop_allin_edge_idx = torch.LongTensor(t_2hop_allin_edge_idx).to(self.device)

    #     t_2hop_edge_idx = torch.cat((t_2hop_sample_edge_idx, t_2hop_allin_edge_idx))
    #     ## the final edge
    #     t_2hop_final_edge_idx_idx = torch.where(torch_np_isin(t_2hop_edge_idx, self.edge_2hop_ord_all, self.device))[0]
        
    #     t_2hop_final_edge_idx = t_2hop_edge_idx[t_2hop_final_edge_idx_idx]
    #     # row is start entities
    #     row = all_head[t_2hop_final_edge_idx]
    #     # col is the edge indices
    #     col = all_tail[t_2hop_final_edge_idx]
    #     vals = [1] * len(row)
    #     ent1st_to_ent2nd_mat = torch.sparse.FloatTensor(torch.stack([row, col], dim=0), torch.FloatTensor(vals).to(self.device), torch.Size((self.n_entities, self.n_entities))).to(self.device)
    #     # 2nd order item2edge
    #     item2entity_2nd = torch.sparse.mm(self.full_relation_i2e_mat, ent1st_to_ent2nd_mat)
    #     item2entity_2nd_indices = item2entity_2nd._indices()

    #     #3rd order item2edge
    #     # seen_entities = torch.cat([torch.arange(self.n_items).to(self.device), row, col, self.row_1st, self.col_1st]).unique()
    #     # unseen_entities = np.setdiff1d(np.arange(self.n_entities), seen_entities.cpu().numpy())
    #     # unseen_entities = torch.LongTensor(unseen_entities).to(self.device)
    #     # ## only search unseen entities to avoid expolsion
    #     # t_3hop_edge_idx = torch.where(torch_np_isin(all_tail, unseen_entities, self.device) & torch_np_isin(all_head, col.unique(), self.device))[0]
    #     # row_3rd = all_head[t_3hop_edge_idx]
    #     # col_3rd = all_tail[t_3hop_edge_idx]
    #     # vals_3rd = [1] * len(row_3rd)
    #     # ent2ent_mat_3rd = torch.sparse.FloatTensor(torch.stack([row_3rd, col_3rd], dim=0), torch.FloatTensor(vals_3rd).to(self.device), torch.Size((self.n_entities, self.n_entities))).to(self.device)
    #     # ent1st_for_ent3rd = torch.sparse.mm(ent1st_to_ent2nd_mat, ent2ent_mat_3rd)
    #     # item2entity_3rd = torch.sparse.mm(self.full_relation_i2e_mat, ent1st_for_ent3rd)
    #     # item2entity_3rd_indices = item2entity_3rd._indices()
        
    #     # row = self.row_1st.to(self.device)
    #     # col = self.col_1st.to(self.device)
    #     # row = torch.cat([self.row_1st, torch.arange(self.n_items).to(self.row_1st.device)]).to(self.device)
    #     # col = torch.cat([self.col_1st, torch.arange(self.n_items).to(self.col_1st.device)]).to(self.device)
    #     # row = torch.cat([self.row_1st, item2entity_2nd_indices[0], item2entity_3rd_indices[0]]).to(self.device)
    #     # col = torch.cat([self.col_1st, item2entity_2nd_indices[1], item2entity_3rd_indices[1]]).to(self.device)

    #     row = torch.cat([self.row_1st, item2entity_2nd_indices[0]]).to(self.device)
    #     col = torch.cat([self.col_1st, item2entity_2nd_indices[1]]).to(self.device)

    #     vals = [1.0] * len(row)
    #     item2ent_graph = torch.sparse.FloatTensor(torch.stack([row, col], dim=0), torch.FloatTensor(vals).to(self.device), torch.Size((self.n_items, self.n_entities))).to(self.device)

    #     return item2ent_graph

    def _edge_sampling(self, edge_index, edge_type, triplet_mask, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        # rebuild graph to only keep the undropped edges
        # item2edge_graph_random_indices = np.where(np.isin(item2edge_graph._indices()[1, :].detach().cpu().numpy(), random_indices))[0]
        # indices = item2edge_graph._indices()[:, item2edge_graph_random_indices]
        # vals = item2edge_graph._values()[item2edge_graph_random_indices]
        # item2edge_graph = torch.sparse.FloatTensor(indices, vals, item2edge_graph.shape).to(item2edge_graph.device)

        return edge_index[:, random_indices], edge_type[random_indices], triplet_mask[random_indices],  random_indices

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))


    def generate_user_specific_mask(self, batch_u2e_mat, user_ids, edge_index, edge_type, random_indices=None, training_epoch=-1):
        """
        
        """
        
        # user_embed = F.normalize(user_embed)
        head, tail = edge_index.detach()
        if training_epoch >= 0 and training_epoch < self.args_config.start_epoch:
            return torch.ones_like(tail), torch.ones((self.n_entities,)).to(self.device)
        user_embed = self.all_embed[user_ids].detach()
        entity_emb = self.all_embed[self.n_users:].detach()

        edge_relation_emb = self.weight[edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        edge_relation_emb = edge_relation_emb.detach()
        # corresponding tail entity * relation 
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb
        # org_edges = self.mlp(torch.cat([self.all_embed[self.n_users:][tail], edge_relation_emb], dim=1))
        # org_edges = F.normalize(org_edges)
        ent_agg = scatter_mean(src=neigh_relation_emb, index=tail, dim_size=self.n_entities, dim=0)
        batch_idx = batch_u2e_mat._indices()
        values = F.relu(torch.cosine_similarity(user_embed[batch_idx[0]], ent_agg[batch_idx[1]])) + 1e-8
        # values = torch.sum(user_embed[batch_idx[0]]*ent_agg[batch_idx[1]], dim=-1)
        # batch_size = 30000000
        # total_step = batch_idx.shape[1] // batch_size + 1
        # all_values = []
        # for i in total_step:
        #     start, end = batch_size * i, batch_size * (i+1) 
        #     values = torch.cosine_similarity(user_embed[batch_idx[0]][start:end], ent_agg[batch_idx[1]][start:end])
        #     all_values.append(values)
        # values = torch.cat(all_values, dim=0)
        # values = torch.sum(torch.mul(user_embed[batch_idx[0]], ent_agg[batch_idx[1]]), axis=1) 
        ent_score = scatter_mean(src=values, index=batch_idx[1], dim_size=batch_u2e_mat.shape[1])
        
        # ent_score_2 = ent_agg[]
        idx = ent_score == 0
        ent_score = ent_score

        # ent_score = (ent_score - ent_score.min()) / (ent_score.max() - ent_score.min()) 
        # divide the N_e
        # batch_u2e_mat = torch.sparse.softmax(batch_u2e_mat, dim=0)

        
        # calculate agged_user - edge scores
        # edge_score = torch.sum(agg_user2edge * org_edges, dim=-1).abs()
        # edge_score = F.sigmoid(F.cosine_similarity(agg_user2edge, org_edges))
        # if training_epoch>=0:
        #     percentile = math.pow(math.e, -self.c * training_epoch)
        # else:
        #     percentile = 0.3
        # max_edge_score = np.percentile(edge_score.detach().cpu().numpy(),  percentile * 100)
        
        # rescore the 0 to 1, in case of the unattached items ( stem from negative sample) cant aggregate the node information 
        
        if self.args_config.score == "mlp":
            triplet_mask = self.score_estimator(entity_emb, head, tail, self.n_entities)
        elif self.args_config.score == "scalar":
            triplet_mask = self.triplet_mask
        triplet_mask = torch.clamp(triplet_mask,0, 1)
        gamma = self.gamma
        # alpha =  self.tau 
        
        
        # edge_score = torch.where(edge_score == 0, 1 - self.triplet_mask, edge_score)
        # if training_epoch >= 0:
            # edge_score[edge_score == 0] += 1
            # edge_score -= torch.rand_like(edge_score) * (1 - math.sqrt(math.sqrt(training_epoch / self.epoch))) * 0.2
        # ent_score = 1 - F.tanh( (1-ent_score) * 5)
        # ent_score = torch.where(ent_score < 0.2, 0, 1)
        final_ent_score = ent_score
        ent_score = ent_score.clone()
        if self.args_config.method == "p":
            pass
            # final_ent_score[idx] += 1e-8
            # final_ent_score[~idx] = final_ent_score[~idx] # * gamma 
            #final_ent_score[~idx] +=  (1-gamma)
        elif self.args_config.method == "g":
            final_ent_score = triplet_mask
        else:
            final_ent_score = final_ent_score * gamma + triplet_mask * (1-gamma)
            # final_ent_score[idx] += triplet_mask[idx]
            # final_ent_score[~idx] = final_ent_score[~idx] * gamma + triplet_mask[~idx] * (1-gamma)

        final_ent_score = torch.clamp(final_ent_score, 0, 1)

        # final_ent_score = ent_score
        
        # if self.args_config.method != "t":
        #     final_ent_score = 1 - F.tanh((1 - ent_score ) * self.beta)

        ## ignore the entities which are unseen in 3-hops
        if training_epoch == -1:
            final_ent_score[idx] = 0

        # ent_score[:self.n_items] = 1
        # edge_score = (ent_score[tail] * ent_score[head] + 1e-10).sqrt() + 1e-10

        # edge_score = 2 * 1 / ( 1 / (ent_score[tail] + 1e-10) + 1 / (ent_score[head] + 1e-10))

        if self.args_config.agg_n2e == "tail":
            edge_score = final_ent_score[tail]
        elif self.args_config.agg_n2e == "mul":
            edge_score = torch.pow(final_ent_score[tail] * final_ent_score[head], 1/2)
        else:
            raise Exception("not supported agg_n2e:%s" % (self.args_config.agg_ne2))
        ## mask zero
        # if training_epoch >= 0 :
        #     props = (1 - edge_score).detach().cpu().numpy()
        #     props = props / np.sum(props)
        #     removed_edge = np.random.choice(range(edge_score.shape[0]), size=edge_score.shape[0] // 10, p=props, replace=False)
        #     edge_score[removed_edge] = 0
        
        ## mask_low_remove
        # if training_epoch >= 0:
        # _, idx = torch.topk(edge_score, k=(edge_score.shape[0] * self.remove_ratio) // 100, largest=False, dim=-1)
        # edge_score[idx] = 0

        # edge_score = torch.where((edge_score - self.tau) < 0, 0, 1)
        # edge_score = edge_score.clamp(0,1)
        return edge_score, final_ent_score
    
    def generate_triplet_mask(self, edge_index, edge_type, training_epoch=-1):
        edge_score = self.triplet_mask
        if training_epoch >= 0:
            edge_score = edge_score + torch.rand_like(edge_score) * (1 - math.sqrt(math.sqrt(training_epoch / self.epoch))) * 0.1
        edge_score = torch.clamp(edge_score, 0, 1) 
        edge_score = F.tanh((edge_score -  torch.clamp(self.tau, 0, 1)) * 10)
        return edge_score

    def forward(self, batch=None, training_epoch=-1):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']
        
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]

        
        # select the users in batch, (tips: sparse matrix do not support index operation)
        # remove the duplicated users
        start_time = time()
        u = np.unique(user.detach().cpu().numpy())
        idx = np.array([[i for i in range(u.shape[0])], u])
        idx = torch.LongTensor(idx).to(self.device)
        v = torch.FloatTensor([1.0] * idx.shape[1]).to(self.device)
        u2batch = torch.sparse.FloatTensor(idx, v, torch.Size((idx.shape[1], self.n_users)))
        # ## shape: (unique_users, edge)
        # batch_u2i_mat = torch.sparse.mm(u2batch, self.item_interact_mat)
        # batch_u2e_mat = torch.sparse.mm(batch_u2i_mat, self.item2ent_graph)
        batch_u2e_mat = torch.sparse.mm(u2batch, self.user2ent_mat)
        edge_index, edge_type, triplet_mask= self.edge_index, self.edge_type, self.triplet_mask
        # triplet_mask = self.generate_triplet_mask(edge_index, edge_type, q_mask, training_epoch=training_epoch)
        
        
        triplet_mask, _ = self.generate_user_specific_mask(batch_u2e_mat, idx[1], edge_index, edge_type, random_indices=None, training_epoch=training_epoch)
        create_mask_time = time() - start_time
        removed_edges = torch.where(triplet_mask > 0)[0]
        edge_index, edge_type, triplet_mask = edge_index[:, removed_edges], edge_type[removed_edges], triplet_mask[removed_edges]
        if self.node_dropout:
            edge_index, edge_type, triplet_mask, random_indices = self._edge_sampling(edge_index, edge_type, triplet_mask, self.node_dropout_rate)
            interact_mat = self._sparse_dropout(self.interact_mat, self.node_dropout_rate)

        # entity_gcn_emb: [n_entity, channel]
        # user_gcn_emb: [n_users, channel]
        entity_gcn_emb, user_gcn_emb = self.gcn(user_emb,
                                                     item_emb,
                                                     self.entity_2nd_emb,
                                                     self.user_2nd_emb,
                                                     edge_index,
                                                     edge_type,
                                                     interact_mat,
                                                     self.weight,
                                                     triplet_mask,
                                                     mess_dropout=self.mess_dropout,
                                                     node_dropout=self.node_dropout,
                                                     training_epoch=training_epoch)

        u_e = user_gcn_emb[user]
        pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item.view(-1)]
        total_loss , mf_loss, emb_loss = self.create_bpr_loss(u_e, pos_e, neg_e) 
        return total_loss, mf_loss, emb_loss, create_mask_time
    def generate(self, eval_epoch=-1):
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        
        edge_index, edge_type, triplet_mask = self.edge_index, self.edge_type, self.triplet_mask
        triplet_mask, _ = self.getmask()
        # triplet_mask = self.generate_triplet_mask(edge_index, edge_type, q_mask)

        return self.gcn(user_emb,
                        item_emb,
                        self.entity_2nd_emb,
                        self.user_2nd_emb,
                        self.edge_index,
                        self.edge_type,
                        self.interact_mat,
                        self.weight,
                        triplet_mask,
                        mess_dropout=False, node_dropout=False,
                        training_epoch=-1)[:3]

    def resample_item2node(self, training_epoch):
        # self.item2ent_graph = self.build_item2entities_graph(self.edge_index)
        pass

    def update_q_mask(self, eval_epoch):
        mask, ent_score = self.getmask()

        self.final_masked_edges = np.intersect1d(self.final_masked_edges, torch.where(mask < self.alpha)[0].cpu().numpy())
        
        self.saved_ent_scores = np.concatenate([self.saved_ent_scores, ent_score.detach().unsqueeze(0).cpu().numpy()], axis=0)

        return self.saved_ent_scores, self.final_masked_edges
        
    def getmask(self):
        user_ids = torch.LongTensor(list(range(self.n_users))).to(self.device)
        batch_u2e_mat = self.user2ent_mat
        # batch_u2i_mat = torch.sparse.mm(user_ids, self.item_interact_mat)
        # batch_u2e_mat = torch.sparse.mm(self.item_interact_mat, self.item2ent_graph)
        edge_index, edge_type, triplet_mask = self.edge_index, self.edge_type, self.triplet_mask
        return self.generate_user_specific_mask(batch_u2e_mat, user_ids, edge_index, edge_type)

    def rating(self, u_g_embeddings, i_g_embeddings):
        # return torch.cosine_similarity(u_g_embeddings.unsqueeze(1), i_g_embeddings.unsqueeze(0), dim=2)
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)
        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # pos_score = torch.cosine_similarity(users, pos_items)
        
        # neg_score = torch.cosine_similarity(users.unsqueeze(1), neg_items.view(batch_size, -1, self.emb_size), dim=2)
        # neg_score = F.relu(neg_score - self.margin)

        # mf_loss = (1 - pos_score) +  torch.sum(neg_score, dim=-1) / (torch.sum(neg_score > 0, dim=-1) + 1e-5)
        # mf_loss = torch.mean(mf_loss)
        

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        return mf_loss + emb_loss , mf_loss, emb_loss
