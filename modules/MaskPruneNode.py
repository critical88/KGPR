'''
Created on July 1, 2020

'''
__author__ = "ffff"

import random
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add, scatter_sum, scatter_max
from torch_scatter.utils import broadcast
from torch_scatter.composite import scatter_softmax
from collections import defaultdict
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
                interact_mat, weight, triplet_mask, q_mask, mess_dropout=True, node_dropout=False, training_epoch=-1):

        entity_res_emb = entity_emb  # [n_entity, channel]
        entity_2nd_res_emb = entity_2nd_emb
        user_res_emb = user_emb   # [n_users, channel]
        user_2nd_res_emb = user_emb

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
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)
            
            # weight = (len(self.convs) - i) / len(self.convs)
            """result emb"""
            # entity_layer_embed.append(entity_emb)
            # user_layer_embed.append(user_emb)
            entity_res_emb = torch.add(entity_res_emb, entity_emb )
            user_res_emb = torch.add(user_res_emb, user_emb)

            # entity_2nd_res_emb = torch.add(entity_2nd_res_emb, entity_2nd_emb)
            # user_2nd_res_emb = torch.add(user_2nd_res_emb, user_2nd_emb)
        if training_epoch >= 0:
            return entity_res_emb, user_res_emb
        else:
            return entity_res_emb, user_res_emb, unmask
        # return torch.cat([entity_res_emb, entity_2nd_res_emb], dim=1), torch.cat([user_res_emb, user_2nd_res_emb], dim=1)


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
        
        self.alpha = args_config.alpha

        self.beta = args_config.beta
        
        self.gamma = args_config.gamma

        self.sample_num = args_config.sample_num

        self.tau = torch.FloatTensor([0.2])

        self.tau = nn.Parameter(self.tau)

        self.margin = 0.5
        self.temperature = 0.2
        self.adj_mat = adj_mat
        self.graph = graph
        self.edge_index, self.edge_type,  = self._get_edges(graph)
        self.need_sampled_entity_idx = None
        self.item2ent_graph = self.build_item2entities_graph(self.edge_index)

        self.edge_cnt = self.edge_index.shape[1]
        
        self.triplet_mask = torch.zeros(self.n_entities)
        
        # self.triplet_mask = torch.zeros(self.edge_index.shape[1]) 

        self.triplet_mask = nn.Parameter(self.triplet_mask)

        ### tail to edge_index map, in order to mapping the user specific mask
        self.final_masked_edges = torch.LongTensor(range(self.edge_index.shape[1]))
        self.saved_ent_scores = np.empty((0, self.n_entities))
        self.q_mask = torch.zeros(self.edge_cnt).to(self.device)
        # # self.q_mask = nn.Parameter(self.q_mask)
        self.q_mask.requires_grad = False
        # prioritize the no mask
        # self.triplet_mask[:, 1] += 1
        
        self._init_weight()
        self._init_user2ent()

        self.all_embed = nn.Parameter(self.all_embed)
        self.weight = nn.Parameter(self.weight)  
        self.gcn = self._init_model()

    def _init_user2ent(self):
        i = self.interact_mat._indices()
        v = torch.FloatTensor([1.0] * i.shape[1]).to(self.device)
        item_interact_mat = torch.sparse.FloatTensor(i, v, torch.Size((self.n_users, self.n_items))).to(self.device)
        ## create user-edge sparse matrix
        user2ent = torch.sparse.mm(item_interact_mat, self.item2ent_graph)
        ## reassign the values to 1
        i = user2ent._indices()
        v = torch.FloatTensor([1.0] * i.shape[1]).to(self.device)
        self.user2ent_mat = torch.sparse.FloatTensor(i, v, user2ent.shape).to(self.device)

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
    
    def build_item2entities_graph(self, edge_index):
        # 1st order
        edge_index = edge_index.t()
        # cache
        if self.need_sampled_entity_idx is None:

            items = edge_index[edge_index[:, 0] < self.n_items][:, 0]
            # item
            self.row_1st = items
            # edge
            self.col_1st = edge_index[edge_index[:, 0] < self.n_items][:, 1]

            # item2edge_graph = torch.sparse.FloatTensor(torch.stack([items, col_1st], dim=0), torch.FloatTensor(vals), torch.Size((self.n_items, len(graph_tensor)))).to(self.device)
            
            # 2nd order 
            # in order to get 2nd order martix, 
            # we first prepared a full relation item-entity matrix, whose value equals 1 if there exists relation between item and entity
            # then we ought to get the 2nd order entity-edge martix, the entity is the tail of 1st order link
            
            # build full relation item-entity matrix
            entities = edge_index[edge_index[:, 0] < self.n_items][:, 1].type(torch.LongTensor).to(self.device)
            vals = [1.0] * len(entities)
            self.full_relation_i2e_mat = torch.sparse.FloatTensor(torch.stack([items, entities], dim=0), torch.FloatTensor(vals).to(self.device), torch.Size((self.n_items, self.n_entities))).to(self.device)
            # build entity-edge matrix
            # the edge must statisfy two criteria:
            # 1. the head of the edge must be the tail of the 1st order edge
            # 2. the head of the edge must have less than 20 edges
            # first we need to find the start_point, i.e. the tail of the 1st order
            start_point = entities.unique()
            # then we get the head node of all edge
            head_node_all_edge = edge_index[:, 0]
            # if head node of the edge is in the start point, then this edge will be filled in the entity-edge matrix 
            self.edge_2nd = np.where(np.isin(head_node_all_edge.detach().cpu().numpy(), start_point.detach().cpu().numpy()))[0]
            #### Attention ####
            #### we omit the overwhelming entity-edge, i.e. the entity has more than 20 edges
            head_entities = edge_index[self.edge_2nd][:, 0]
            bincount = head_entities.bincount(minlength=self.n_entities)
            self.saved_entities_idx = torch.where(bincount < 10)[0]
            self.need_sampled_entity_idx = torch.where(bincount >= 10)[0]

        
        head_node_all_edge = edge_index[:, 0]
        sample_num = self.sample_num
        sampled_edge_idx = np.empty(0, dtype=np.int32)
        if self.args_config.is_two_hop:
            for entity_id in self.need_sampled_entity_idx:
                entity_edges = torch.where(head_node_all_edge == entity_id)[0]
                entity_edges = np.intersect1d(entity_edges.cpu().numpy(), self.edge_2nd)
                entity_edges = np.random.choice(entity_edges, sample_num, replace=False)
                sampled_edge_idx = np.concatenate([sampled_edge_idx, entity_edges])
            
        ## edge_idx exclude the nodes that link more than 10 edges
        edge_idx = np.where(np.isin(head_node_all_edge.detach().cpu().numpy(), self.saved_entities_idx.detach().cpu().numpy()))[0]
        # the nodes that has more than 10 edges
        edge_idx = np.concatenate((sampled_edge_idx, edge_idx))
        ## the final edge
        final_edge_idx = np.intersect1d(edge_idx, self.edge_2nd)
        
        # row is start entities
        row = edge_index[final_edge_idx][:, 0]
        # col is the edge indices
        col = edge_index[final_edge_idx][:, 1]
        vals = [1.] * len(row)
        ent2ent_mat = torch.sparse.FloatTensor(torch.stack([row, col], dim=0), torch.FloatTensor(vals).to(self.device), torch.Size((self.n_entities, self.n_entities))).to(self.device)
        # 2nd order item2edge
        item2entity_2nd = torch.sparse.mm(self.full_relation_i2e_mat, ent2ent_mat)
        indices = item2entity_2nd._indices()
        # row = row_1st.to(self.device)
        # col = col_1st.to(self.device)
        row = torch.cat((self.row_1st.to(self.device), indices[0]) ,dim=0)
        col = torch.cat((self.col_1st.to(self.device), indices[1]), dim=0)
        vals = [1.0] * len(row)
        item2ent_graph = torch.sparse.FloatTensor(torch.stack([row, col], dim=0), torch.FloatTensor(vals).to(self.device), torch.Size((self.n_items, self.n_entities))).to(self.device)

        return item2ent_graph

    def _edge_sampling(self, edge_index, edge_type, triplet_mask, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)

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


    def generate_user_specific_mask(self, batch_u2e_mat, user_embed, edge_index, edge_type, random_indices=None, training_epoch=-1):
        """
        
        """
        # user_embed = F.normalize(user_embed)
        head, tail = edge_index.detach()
        user_embed = user_embed.detach()
        edge_relation_emb = self.weight[edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        # corresponding tail entity * relation 
        org_edges = self.all_embed[self.n_users:][tail] * edge_relation_emb
        org_edges = org_edges.detach()
        tail_ent = scatter_mean(src=org_edges, index=tail, dim_size=self.n_entities, dim=0)
        batch_idx = batch_u2e_mat._indices()
        values = torch.cosine_similarity(user_embed[batch_idx[0]], tail_ent[batch_idx[1]]).abs()
        ent_score = scatter_mean(src=values, index=batch_idx[1], dim_size=batch_u2e_mat.shape[1])
        idx = ent_score == 0
        triplet_mask = 1 - self.triplet_mask
        gamma = self.gamma
        
        if self.args_config.method == "pg":
            ent_score[idx] += triplet_mask[idx]
            ent_score[~idx] = ent_score[~idx] * gamma + triplet_mask[~idx] * (1-gamma)
            
        elif self.args_config.method == "p":
            ent_score[idx] += 1
            ent_score[~idx] = ent_score[~idx] * gamma 
            ent_score[~idx] +=  (1-gamma)
        elif self.args_config.method == "g":
            ent_score = triplet_mask

        ent_score = torch.clamp(ent_score, 0, 1)
        ent_score = 1 - F.tanh((1 - ent_score ) * self.beta)
        edge_score = ent_score[tail]
        # edge_score = torch.where((edge_score - self.tau) < 0, 0, 1)
        # edge_score = edge_score.clamp(0,1)
        return edge_score, ent_score
    
    def generate_triplet_mask(self, edge_index, edge_type, q_mask, training_epoch=-1):
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
        u = np.unique(user.detach().cpu().numpy())
        idx = np.array([[i for i in range(u.shape[0])], u])
        
        idx = torch.LongTensor(idx).to(self.device)
        v = torch.FloatTensor([1.0] * idx.shape[1]).to(self.device)
        u2batch = torch.sparse.FloatTensor(idx, v, torch.Size((idx.shape[1], self.n_users)))
        
        # ## shape: (unique_users, edge)
        batch_u2e_mat = torch.sparse.mm(u2batch, self.user2ent_mat)
        edge_index, edge_type, triplet_mask, q_mask = self.edge_index, self.edge_type, self.triplet_mask, self.q_mask
        
        
        triplet_mask, _ = self.generate_user_specific_mask(batch_u2e_mat, user_emb[idx[1]], edge_index, edge_type, random_indices=None, training_epoch=training_epoch)
        if self.node_dropout:
            edge_index, edge_type, triplet_mask, random_indices = self._edge_sampling(self.edge_index, self.edge_type, triplet_mask, self.node_dropout_rate)
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
                                                     q_mask,
                                                     mess_dropout=self.mess_dropout,
                                                     node_dropout=self.node_dropout,
                                                     training_epoch=training_epoch)

        u_e = user_gcn_emb[user]
        pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item.view(-1)]
        total_loss , mf_loss, emb_loss = self.create_bpr_loss(u_e, pos_e, neg_e) 
        return total_loss, mf_loss, emb_loss
    def generate(self, eval_epoch=-1):
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        
        edge_index, edge_type, triplet_mask, q_mask = self.edge_index, self.edge_type, self.triplet_mask, self.q_mask
        triplet_mask, _ = self.getmask()

        return self.gcn(user_emb,
                        item_emb,
                        self.entity_2nd_emb,
                        self.user_2nd_emb,
                        self.edge_index,
                        self.edge_type,
                        self.interact_mat,
                        self.weight,
                        triplet_mask,
                        self.q_mask,
                        mess_dropout=False, node_dropout=False,
                        training_epoch=-1)[:3]

    def update_q_mask(self, eval_epoch):
        """
        save the mask and scores in each epoch
        "final_masked_edges", the mask file , is default to threshold 0.05
        "_saved_ent_scores" the scores in each epoch, used to remove nodes by different methods
        """
        mask, ent_score = self.getmask()

        self.final_masked_edges = np.intersect1d(self.final_masked_edges, torch.where(mask < self.alpha)[0].cpu().numpy())

        self.saved_ent_scores = np.concatenate([self.saved_ent_scores, ent_score.detach().unsqueeze(0).cpu().numpy()], axis=0)
        np.save(f"{self.args_config.data_path}{self.args_config.dataset}/{self.args_config.model}_{self.args_config.kg_file}_{self.args_config.is_two_hop}_{self.args_config.method}_saved_ent_scores.npy", self.saved_ent_scores)
        np.save(f"{self.args_config.data_path}{self.args_config.dataset}/{self.args_config.model}_{self.args_config.kg_file}_{self.args_config.is_two_hop}_{self.args_config.method}_final_masked_edges.npy", self.final_masked_edges)

    def getmask(self):
        edge_index, edge_type, triplet_mask, q_mask = self.edge_index, self.edge_type, self.triplet_mask, self.q_mask
        return self.generate_user_specific_mask(self.user2ent_mat, self.all_embed[:self.n_users], edge_index, edge_type)

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)
        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))


        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        return mf_loss + emb_loss , mf_loss, emb_loss
