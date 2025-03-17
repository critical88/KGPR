'''
Created on July 1, 2020
@author: Tinglin Huang (tinglin.huang@zju.edu.cn)
'''
__author__ = "huangtinglin"

import random

import json
import torch
import numpy as np
import os
from time import time
from prettytable import PrettyTable
from tqdm import tqdm
from utils.parser import parse_args
from utils.data_loader import load_data

from utils.evaluate import test, evaluate, save_unpruned_node, get_orginal_kg, get_masked_info
from utils.helper import early_stopping
import pandas as pd
n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0



# def get_feed_data(train_entity_pairs, train_user_set, n_items, n_users):
#     num_neg_sample = args.num_neg_sample
#     def negative_sampling(user_item, train_user_set):
#         neg_items = list()
#         for user, _ in user_item.cpu().numpy():
#             user = int(user)
#             each_negs = list()
#             neg_item = np.random.randint(low=0, high=n_items, size=num_neg_sample)
#             if len(set(neg_item) & set(train_user_set[user]))==0:
#                 each_negs += list(neg_item)
#             else:
#                 neg_item = list(set(neg_item) - set(train_user_set[user]))
#                 each_negs += neg_item
#                 while len(each_negs)<num_neg_sample:
#                     n1 = np.random.randint(low=0, high=n_items, size=1)[0]
#                     if n1 not in train_user_set[user]:
#                         each_negs += [n1]
#             neg_items.append(each_negs)

#         return neg_items
#     feed_dict = {}
#     entity_pairs = train_entity_pairs
#     feed_dict['users'] = entity_pairs[:, 0]
#     feed_dict['pos_items'] = entity_pairs[:, 1]
#     feed_dict['neg_items'] = torch.LongTensor(negative_sampling(entity_pairs,train_user_set))
#     # feed_dict['neg_users'] = torch.LongTensor(negative_sampling_user(entity_pairs,train_user_set))
#     return feed_dict
def get_feed_dict(train_entity_pairs, start, end, train_user_set, n_items):

    def negative_sampling(user_item, train_user_set, n):
        neg_items = []
        # cand_rands = torch.randint(0, n_items, (user_item.shape[0], n)).numpy()
        for i, (user, _) in enumerate(user_item.cpu().numpy()):
            user = int(user)
            negitems = []
            # negitems = list(np.setdiff1d(cand_rands[i], train_user_set[user]))
            while len(negitems) < n:  # sample n times
                while True:
                    negitem = random.choice(range(n_items))
                    if negitem not in train_user_set[user]:
                        break
                negitems.append(negitem)
            neg_items.append(negitems)
        return neg_items
    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end].to(device)
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(negative_sampling(entity_pairs,
                                                                train_user_set, 1)).to(device)
    return feed_dict

def save_args_config(args):
    saved_config_file = os.path.join(args.save_dir, "config.json")
    arg_dict = json.dumps(args.__dict__)
    with open(saved_config_file, "w") as f:
        f.write(arg_dict)

def overwrite_config(args):
    saved_config_file = os.path.join(args.save_dir, "config.json")
    if not os.path.exists(saved_config_file):
        return args
    with open(saved_config_file) as f:
        arg_org = json.loads(f.read())
    args.model = arg_org["model"]
    args.is_two_hop = arg_org["is_two_hop"]
    args.dataset = arg_org["dataset"]
    args.kg_file = arg_org["kg_file"]
    args.method = arg_org["method"]
    args.agg_n2e = arg_org["agg_n2e"]
    args.num_sample_user2ent = arg_org["num_sample_user2ent"]
    args.score = arg_org["score"] if "score" in arg_org else "scalar"
    return args

def train(args):
    global device
    """fix the random seed"""
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_args_config(args)
    
    device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")

    """build dataset"""
    train_cf, test_cf, user_dict, n_params, graph, mat_list = load_data(args)
    adj_mat_list, norm_mat_list, mean_mat_list = mat_list

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']

    """cf data"""
    train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in test_cf], np.int32))

    """define model"""
    
    if args.model == "mask_node_final":
        from modules.MaskPruneNodeFinal import Recommender
        model = Recommender(n_params, args, graph, mean_mat_list[0]).to(device)
    
    else:
        raise Exception(f"do not support this model:{args.model}")
    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False

    

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    print("start training ...")
    
    for epoch in range(args.epoch):
        
        """training CF"""
        # # shuffle training data
        index = np.arange(len(train_cf))
        np.random.shuffle(index)
        train_cf_pairs = train_cf_pairs[index]
        
        ## shuffle training data
        # train_data =  remap_train(user_dict['train_user_set'], user_dict['train_item_set'], n_users, n_items)
        # index = np.arange(len(train_data))
        # np.random.shuffle(index)
        # train_cf_pairs = torch.LongTensor(train_data[index]).to(device)
        # """training"""
        loss, s = 0, 0
        train_s_t = time()
        total_score_time = 0
        while s + args.batch_size <= len(train_cf_pairs):
            batch = {}
            # entity_pairs = train_cf_pairs[s: s + args.batch_size]
            # batch['users'] = entity_pairs[:, 0]
            # batch['pos_items'] = entity_pairs[:, 1]
            # batch['neg_items'] = entity_pairs[:, 2]
            batch = get_feed_dict(train_cf_pairs,
                                  s, s + args.batch_size,
                                  user_dict['train_user_set'], n_items)
            batch_loss, _, _, score_time = model(batch, epoch)
            total_score_time += score_time
            batch_loss = batch_loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            s += args.batch_size

        """training CF"""
        # shuffle training data
        # if epoch % 5 == 0:
        #     index = np.arange(len(train_cf))
        #     np.random.shuffle(index)
        #     train_cf_pairs = train_cf_pairs[index]
        #     print("start prepare feed data...")
        #     all_feed_data = get_feed_data(train_cf_pairs, user_dict['train_user_set'], n_items, n_users)  # {'user': [n,], 'pos_item': [n,], 'neg_item': [n, n_sample]}

        """training"""
        # loss = 0
        # train_s_t = time()
        # n_batch = len(train_cf) // args.batch_size + 1
        # for i in tqdm(range(n_batch), desc="epoch:{}, training".format(epoch + 1)):
        #     batch = dict()
        #     batch['users'] = all_feed_data['users'][i*args.batch_size:(i+1)*args.batch_size].to(device)
        #     batch['pos_items'] = all_feed_data['pos_items'][i*args.batch_size:(i+1)*args.batch_size].to(device)
        #     batch['neg_items'] = all_feed_data['neg_items'][i*args.batch_size:(i+1)*args.batch_size,:].to(device)
        #     batch_loss, _, _ = model(batch)

        #     batch_loss = batch_loss
        #     optimizer.zero_grad()
        #     batch_loss.backward()
        #     optimizer.step()

        #     loss += batch_loss

        train_e_t = time()
        
        if args.model.__contains__("mask") and ( epoch >= 0):
            ret = model.update_q_mask(epoch)
            if ret is not None:
                saved_ent_scores, final_masked_edges = ret
                np.save(os.path.join(save_dir, "saved_ent_scores.npy"), saved_ent_scores)
                np.save(os.path.join(save_dir, "final_masked_edges.npy"), final_masked_edges)
        
        
        if (epoch+1) % 10 == 0 or epoch == 1  :
            """testing"""
            test_s_t = time()
            ret = test(model, user_dict, n_params, epoch)
            # ret = evaluate(model, 2048, n_items, user_dict["train_user_set"], user_dict["test_user_set"], device)
            test_e_t = time()

            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time", "tesing time", "Loss", "recall", "ndcg", "precision"]
            # ret = ret[20]
            train_res.add_row(
                [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), ret['recall'], ret['ndcg'], ret['precision']]
            )
            print(train_res, flush=True)

            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
            cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=10)
            if should_stop:
                break

            """save weight"""
            if ret['recall'][0] == cur_best_pre_0 and args.save:

                torch.save(model.state_dict(), os.path.join(save_dir, "checkpoint.ckpt"))
        else:
            # logging.info('training loss at epoch %d: %f' % (epoch, loss.item()))
            print('using time %.4f=%.4f+%.4f, training loss at epoch %d: %.4f' % (train_e_t - train_s_t, train_e_t - train_s_t- total_score_time, total_score_time, epoch, loss.item()), flush=True)
        
    
    if args.model.__contains__("mask"):
        recreate_kgfile_by_percentile(model, args, save_dir)
    else:
        remove_IMP_nodes(model, args, save_dir)
    print('early stopping at %d, recall@20:%.4f' % (epoch, cur_best_pre_0))

def predict(args):
    global device
    if args.pretrain_model_path:
        args.save_dir = args.pretrain_model_path

    args = overwrite_config(args)
    device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")

    """build dataset"""
    train_cf, test_cf, user_dict, n_params, graph, mat_list = load_data(args)
    adj_mat_list, norm_mat_list, mean_mat_list = mat_list


    """cf data"""
    train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in test_cf], np.int32))

    if args.model == "mask_node_final":
        from modules.MaskPruneNodeFinal import Recommender
        model = Recommender(n_params, args, graph, mean_mat_list[0]).to(device)
    else:
        raise Exception("unknow model")
    model = load_model(model, args.pretrain_model_path)
    # recreate_kgfile_by_threshold(model, args)
    if args.model.__contains__("mask"):
        recreate_kgfile_by_percentile(model, args)
        # remove_nodes(model, args)
    else:
        remove_IMP_nodes(model, args)
    saved_path = os.path.join(args.data_dir, "result", "{}_{}_{}_norm_sig".format(args.dataset, args.num_neg_sample, args.kg_file))
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    

    test_s_t = time()
    ret = test(model, user_dict, n_params)
    test_e_t = time()

    train_res = PrettyTable()
    train_res.field_names = [ "tesing time", "recall", "ndcg", "precision", "hit_ratio"]
    train_res.add_row(
        [test_e_t - test_s_t, ret['recall'], ret['ndcg'], ret['precision'], ret['hit_ratio']]
    )
    print(train_res)
    # np.save(os.path.join(saved_path, "cf_score.npy"), np.array(ret["ratings"]))

def remove_IMP_nodes(model, args, save_dir=None):
    save_dir = save_dir if save_dir is not None else args.pretrain_model_path
    n_relation = (model.n_relations - 1) / 2 - 1

    head, tail = model.edge_index
    edge_type = model.edge_type - 1

    can_triplets_np = get_orginal_kg(model)
    idx = edge_type <= n_relation
    edge_embd = model.all_embed[model.n_users:][tail[idx]] ## * model.gcn.weight[edge_type[idx]]
    
    edge_norm = edge_embd.norm(dim=-1).detach().cpu().numpy()
    
    thresholds = [10,  30,  50,  70,  90, 95, 98]
    save_kg_dir = os.path.join(save_dir, "percentile")
    if not os.path.exists(save_kg_dir):
        os.makedirs(save_kg_dir)
    for threshold in thresholds:
        save_unpruned_file = os.path.join(save_kg_dir, f"{args.dataset}_kgrs_{threshold}.txt")
        
        saved_kg_idx = np.argsort(-edge_norm)[:len(edge_norm) * (100 - threshold) // 100]
        # removed_threshold = np.percentile(mean_scores, percentile).round(3)

        # saved_triplets = can_triplets_np[mean_scores > removed_threshold]
        saved_triplets = can_triplets_np[saved_kg_idx]

        np.savetxt(save_unpruned_file, saved_triplets, fmt='%i')
    
def remove_nodes(model, args):
    print("start pruning")
    start = time()
    mask_file = f"{args.data_path}{args.dataset}/{args.model}_{args.kg_file}_final_masked_edges_{args.is_two_hop}.npy"
    save_unpruned_file = f"{args.data_path}{args.dataset}/{args.model}_{args.method}_{args.kg_file}_kg_pruned_{args.is_two_hop}.txt"
    save_unpruned_node(model, mask_file, save_unpruned_file)
    print("end pruning, consume:" + str(time() - start))

def recreate_kgfile_by_threshold(model, args, save_dir=None):
    save_dir = save_dir if save_dir is not None else args.pretrain_model_path
    saved_ent_scores = os.path.join(save_dir, "saved_ent_scores.npy")
    head, tail = model.edge_index
    ent_scores = np.load(saved_ent_scores)
    for threshold in [0.01, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
        save_kg_dir = os.path.join(save_dir, "threshold")
        if not os.path.exists(save_kg_dir):
            os.makedirs(save_kg_dir)
        save_kg_file = os.path.join(save_kg_dir, f"{threshold}_kg_pruned.txt")
        
        all_mask = ent_scores < threshold
        final_mask = all_mask[0]
        for i in range(1, ent_scores.shape[0]):
            final_mask = final_mask & all_mask[i]
        edge_mask = final_mask[tail.detach().cpu().numpy()]
        masked_triplest_df, all_triplets_df = get_masked_info(model, edge_mask)
    
        intersected_triplets = all_triplets_df[~(all_triplets_df["match"].isin(masked_triplest_df["match"]))]
        intersected_triplets[["h", "r", "t"]].to_csv(save_kg_file, index=False, header=None, sep=" ")

def epoch_mean(args, ent_scores, model):

    epochs= [5]

    for epoch in epochs:
        yield ent_scores[args.start_epoch:epoch].mean(axis=0), f"{epoch}"
        if epoch == 5:
            yield ent_scores[epoch], f"{epoch}_str"

def epoch_last(args, ent_scores, model):
    return [(ent_scores[-1], "last_epoch")]

def model_mask(args, ent_scores, model):

    return [(None, "mask")]
def recreate_kgfile_by_percentile(model, args, save_dir=None):
    save_dir = save_dir if save_dir is not None else args.pretrain_model_path
    saved_ent_scores = os.path.join(save_dir, "saved_ent_scores.npy")
    # epochs = [5, 2000]
    
    head, tail = model.edge_index
    ent_scores = np.load(saved_ent_scores)
    n_relation = (model.n_relations - 1) / 2 - 1
    edge_type = model.edge_type - 1
    org_index = torch.where(edge_type <= n_relation)[0]
    
    entity_emb = model.all_embed[model.n_users:]
    edge_relation_emb = model.weight[edge_type]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
    # corresponding tail entity * relation 
    neigh_relation_emb = entity_emb[tail] * edge_relation_emb
    # org_edges = self.mlp(torch.cat([self.all_embed[self.n_users:][tail], edge_relation_emb], dim=1))
    # org_edges = F.normalize(org_edges)
    from torch_scatter import scatter_mean
    ent_agg = scatter_mean(src=neigh_relation_emb, index=tail, dim_size=model.n_entities, dim=0)

    ops = ["epoch", "last"]
    tail_index = tail[org_index].detach().cpu().numpy()
    head_index = head[org_index].detach().cpu().numpy()
    tail = tail[org_index]
    head = head[org_index]
    methods =  ["sum",  "mul"]
    can_triplets_np = get_orginal_kg(model)
    for method_name in methods:
        if method_name == "model":
            ops = ["model"]

        for op in ops:
            if op == "epoch":
                method = epoch_mean
            elif op == "last":
                method = epoch_last
            elif op == "model":
                method = model_mask
            else:
                raise Exception("op error")
            
            for mean_scores, name in method(args, ent_scores, model):
                
                if method_name.__contains__("model"):
                    triplet_mask, _  = model.getmask()
                    edge_scores = triplet_mask[org_index].detach().cpu().numpy()

                elif method_name.__contains__("agg"):
                    personal_score = mean_scores[tail_index]
                    local_score = (1 - torch.cosine_similarity(ent_agg[head], ent_agg[tail]).detach().cpu().numpy()) / 2
                    # edge_scores = np.sqrt(personal_score * local_score)
                    # edge_scores = 2 * 1 / ( 1 / (personal_score + 1e-10) + 1 / (local_score + 1e-10))
                    edge_scores = np.min([personal_score, local_score], axis=0)
                elif method_name == "sum":
                    head_score, tail_score = mean_scores[head_index], mean_scores[tail_index]
                    edge_scores = []
                    # 0 denotes only consider the tail score
                    for i in [0]:
                        edge_scores.append((i * head_score + (1 - i) * tail_score, "beta" + str(i)))
                elif method_name == "min":
                    edge_scores = np.min([mean_scores[tail_index], mean_scores[head_index]], axis=0)
                elif method_name == "mul":
                    edge_scores = []
                    head_score, tail_score = mean_scores[head_index], mean_scores[tail_index]
                    for i in [ 0.5]:
                        edge_scores.append(((head_score** i) * (tail_score** (1 - i)) , "beta" + str(i)))
                    # edge_scores = mean_scores[tail_index] * mean_scores[head_index]
                else:
                    raise Exception("not support method name")
                # edge_scores = 2 * 1 / ( 1 / (mean_scores[tail_index] + 1e-10) + 1 / (mean_scores[head_index] + 1e-10))
                # edge_scores = np.sqrt(mean_scores[tail_index] * mean_scores[head_index])
                # edge_scores = np.min([mean_scores[tail[org_index].detach().cpu().numpy()], mean_scores[head[org_index].detach().cpu().numpy()]], axis=0)
                
                save_kg_dir = os.path.join(save_dir, f"{method_name}_{name}")
                if not os.path.exists(save_kg_dir):
                    os.makedirs(save_kg_dir)
                
                if type(edge_scores) != list:
                    edge_scores = [(edge_scores, "n")]
                for edge_score, edge_name in edge_scores:
                    for percentile in [10,20,30, 40, 50, 60, 70, 80, 90, 95, 98]:
                        save_kg_file = os.path.join(save_kg_dir, f"{args.dataset}_{method_name}_kgpr_percentile{percentile}_{name}_{edge_name}.txt")
                        saved_kg_idx = np.argsort(-edge_score)[:len(edge_score) * (100 - percentile) // 100]
                        saved_triplets = can_triplets_np[saved_kg_idx]
                        np.savetxt(save_kg_file, saved_triplets, fmt='%i')
                    if args.dataset in ["alibaba-fashion", "amazon-book", "last-fm"]:
                        min_score = edge_score[edge_score > 0].min()
                        edge_score = (edge_score - min_score) / (edge_score.max()  - min_score)
                    for th in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]:
                        save_kg_file = os.path.join(save_kg_dir, f"{args.dataset}_{method_name}_kgpr_th_{th}_{name}_{edge_name}.txt")
                        saved_kg_idx = np.where(edge_score >= th)[0]
                        saved_triplets = can_triplets_np[saved_kg_idx]
                        np.savetxt(save_kg_file, saved_triplets, fmt='%i')
        
    
def load_model(model, model_path):
    checkpoint = torch.load(os.path.join(model_path, "checkpoint.ckpt"), map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()
    return model

if __name__ == '__main__':

    """read args"""
    global args
    args = parse_args()
    save_dir = "model_{}_{}_{}_{}_{}_{}_{}_triplet0".format(args.model, args.score,  args.dataset, args.gamma, args.method, args.is_two_hop,  args.num_sample_user2ent)
    save_dir = os.path.join(args.out_dir,save_dir)
    args.save_dir = save_dir
    if not args.pretrain_model_path:
        print("training")
        train(args) 
    else:
        print("predicting")
        predict(args)
    
    
