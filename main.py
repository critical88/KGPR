'''
Created on July 1, 2020
@author: Tinglin Huang (tinglin.huang@zju.edu.cn)
'''
__author__ = "huangtinglin"

import random

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

def get_feed_dict(train_entity_pairs, start, end, train_user_set, n_items):

    def negative_sampling(user_item, train_user_set, n):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            negitems = []
            for i in range(n):  # sample n times
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

def train():
    """fix the random seed"""
    seed = 2021
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device
    args = parse_args()
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
    if args.model == "normal":
        from modules.Normal import Recommender
        model = Recommender(n_params, args, graph, mean_mat_list[0]).to(device)
    elif args.model == "mask_node":
        from modules.MaskPruneNode import Recommender
        model = Recommender(n_params, args, graph, mean_mat_list[0]).to(device)
    else:
        raise Exception(f"do not support this model:{args.model}")
    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False
    saved_path = os.path.join(args.data_dir, "result", args.dataset)
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    print("start training ...")
    for epoch in range(args.epoch):

        
        """training CF"""
        # # shuffle training data
        index = np.arange(len(train_cf))
        np.random.shuffle(index)
        train_cf_pairs = train_cf_pairs[index]
        
        # """training"""
        loss, s = 0, 0
        train_s_t = time()
        while s + args.batch_size <= len(train_cf_pairs):
            batch = {}
            batch = get_feed_dict(train_cf_pairs,
                                  s, s + args.batch_size,
                                  user_dict['train_user_set'], n_items)
            batch_loss, _, _ = model(batch, epoch)

            batch_loss = batch_loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            s += args.batch_size


        train_e_t = time()
        
        if args.model.__contains__("mask") and ( epoch >= 0):
            model.update_q_mask(epoch)
        
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
                torch.save(model.state_dict(), os.path.join(args.out_dir, "model_{}_{}_{}_{}_{}_{}_{}.ckpt".format(args.model,args.dataset, args.num_neg_sample, args.kg_file, args.method, args.is_two_hop, args.sample_num)))
        else:
            # logging.info('training loss at epoch %d: %f' % (epoch, loss.item()))
            print('using time %.4f, training loss at epoch %d: %.4f' % (train_e_t - train_s_t, epoch, loss.item()), flush=True)
    
    if args.model.__contains__("mask"):
        remove_nodes(model, args)
    else:
        remove_IMP_nodes(model, args)
    print('early stopping at %d, recall@20:%.4f' % (epoch, cur_best_pre_0))

def predict():
    global args, device
    args = parse_args()
    device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")

    """build dataset"""
    train_cf, test_cf, user_dict, n_params, graph, mat_list = load_data(args)
    adj_mat_list, norm_mat_list, mean_mat_list = mat_list


    """cf data"""
    train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in test_cf], np.int32))

    """define model"""
    if args.model == "normal":
        from modules.Normal import Recommender
        model = Recommender(n_params, args, graph, mean_mat_list[0]).to(device)
    elif args.model == "mask_node":
        from modules.MaskPruneNode import Recommender
        model = Recommender(n_params, args, graph, mean_mat_list[0]).to(device)
    else:
        raise Exception("unknow model")
    model = load_model(model, args.pretrain_model_path)

    recreate_kgfile_by_threshold(model, args)
    # recreate_kgfile_by_percentile(model, args)
    # if args.model.__contains__("mask"):
    #     remove_nodes(model, args)
    # else:
        # remove_IMP_nodes(model, args)
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

def remove_IMP_nodes(model, args):
    """
    used in normal modules

    remove nodes whose norm lives in bottom 30% 
    """
    
    n_relation = (model.n_relations - 1) / 2 - 1

    head, tail = model.edge_index
    edge_type = model.edge_type - 1

    can_triplets_np = get_orginal_kg(model)
    idx = edge_type <= n_relation
    edge_embd = model.all_embed[model.n_users:][tail[idx]] ## * model.gcn.weight[edge_type[idx]]
    
    edge_norm = edge_embd.norm(dim=-1).detach().cpu().numpy()
    
    thresholds = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    for threshold in thresholds:
        save_unpruned_file = f"{args.data_path}{args.dataset}/{args.model}_{args.method}_{args.kg_file}_{threshold}_node_kg_pruned.txt"
        
        removed_threshold = np.percentile(edge_norm, threshold).round(3)

        saved_triplets = can_triplets_np[edge_norm > removed_threshold]

        np.savetxt(save_unpruned_file, saved_triplets, fmt='%i')
    
def remove_nodes(model, args):
    """
    remove nodes by saved mask
    """
    print("start pruning")
    start = time()
    mask_file = f"{args.data_path}{args.dataset}/{args.model}_{args.kg_file}_final_masked_edges_{args.is_two_hop}.npy"
    save_unpruned_file = f"{args.data_path}{args.dataset}/{args.model}_{args.method}_{args.kg_file}_kg_pruned_{args.is_two_hop}.txt"
    save_unpruned_node(model, mask_file, save_unpruned_file)
    print("end pruning, consume:" + str(time() - start))

def recreate_kgfile_by_threshold(model, args):
    """
    remove nodes by specified threshold
    """
    saved_ent_scores = f"{args.data_path}{args.dataset}/{args.model}_{args.kg_file}_{args.is_two_hop}_{args.method}_saved_ent_scores.npy"
    head, tail = model.edge_index
    ent_scores = np.load(saved_ent_scores)
    for threshold in [0.01, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:

        save_kg_file = f"{args.data_path}{args.dataset}/threshold/{args.model}_{args.method}_{threshold}_kg_pruned_{args.is_two_hop}.txt"
        if not os.path.exists(f"{args.data_path}{args.dataset}/threshold/"):
            os.makedirs(f"{args.data_path}{args.dataset}/threshold/")
        all_mask = ent_scores < threshold
        final_mask = all_mask[0]
        for i in range(1, ent_scores.shape[0]):
            final_mask = final_mask & all_mask[i]
        edge_mask = final_mask[tail.detach().cpu().numpy()]
        masked_triplest_df, all_triplets_df = get_masked_info(model, edge_mask)
    
        intersected_triplets = all_triplets_df[~(all_triplets_df["match"].isin(masked_triplest_df["match"]))]
        intersected_triplets[["h", "r", "t"]].to_csv(save_kg_file, index=False, header=None, sep=" ")

def recreate_kgfile_by_percentile(model, args):
    """
    remove nodes by percentile
    """
    saved_ent_scores = f"{args.data_path}{args.dataset}/{args.model}_{args.kg_file}_{args.is_two_hop}_{args.method}_saved_ent_scores.npy"
    head, tail = model.edge_index
    ent_scores = np.load(saved_ent_scores)
    mean_scores = ent_scores.mean(axis=0)
    
    n_relation = (model.n_relations - 1) / 2 - 1
    edge_type = model.edge_type - 1

    org_index = torch.where(edge_type <= n_relation)[0]
    tail = tail[org_index]
    mean_scores = mean_scores[tail.detach().cpu().numpy()]

    can_triplets_np = get_orginal_kg(model)

    if not os.path.exists(f"{args.data_path}{args.dataset}/percentile/"):
            os.makedirs(f"{args.data_path}{args.dataset}/percentile/")
    for percentile in [10, 20,30, 40,50, 60,70, 80, 90]:

        save_kg_file = f"{args.data_path}{args.dataset}/percentile/{args.model}_{args.method}_{percentile}_kg_pruned_{args.is_two_hop}.txt"
        
        removed_threshold = np.percentile(mean_scores, percentile).round(3)

        saved_triplets = can_triplets_np[mean_scores > removed_threshold]

        np.savetxt(save_kg_file, saved_triplets, fmt='%i')
        
    
def load_model(model, model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()
    return model

if __name__ == '__main__':
    train() 
    # predict()
