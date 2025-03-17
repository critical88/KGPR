import argparse


def parse_args():
    data_dir = ""
    parser = argparse.ArgumentParser(description="KGPR")

    # ===== dataset ===== #
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--dataset", nargs="?", default="last-fm", help="Choose a dataset:[last-fm,amazon-book,alibaba]")
    parser.add_argument(
        "--data_path", nargs="?", default="data/", help="Input data path."
    )
    parser.add_argument("--remove_ratio", default=30, type=int)
    parser.add_argument("--model", default="mask_node_test", help="choose in [kgin, lightgcn, mask, mask_node]")
    parser.add_argument("--num_neg_sample", default=1, type=int)
    parser.add_argument("--data_dir", default=data_dir)
    parser.add_argument("--pretrain_model_path", default="")
    parser.add_argument("--alpha", type=float, default=0.05, help="the threshold to judge whether remove the node, it only work when creating pruned kg")
    parser.add_argument("--beta", type=float, default=5, help="1 - tanh(beta * (1 - x))")
    parser.add_argument("--gamma", type=float, default=0.5, help="the cofficient to balance the global and personal influence")
    
    parser.add_argument("--method", type=str, default="pg", help="albation study, p denotes the personalize, g denotes global")
    parser.add_argument("--is_two_hop", action="store_false", default=True)
    parser.add_argument("--sample_num", type=int, default=10)
    
    # ===== train ===== #
    parser.add_argument('--epoch', type=int, default=2000, help='number of epochs')
    parser.add_argument('--sparsity', type=float, default=0.5, help='sparsity')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--dim', type=int, default=64, help='embedding size')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--sim_regularity', type=float, default=1e-4, help='regularization weight for latent factor')
    parser.add_argument("--inverse_r", action="store_false", default=True, help="consider inverse relation or not")
    parser.add_argument("--node_dropout", type=bool, default=True, help="consider node dropout or not")
    parser.add_argument("--node_dropout_rate", type=float, default=0.5, help="ratio of node dropout")
    parser.add_argument("--mess_dropout", type=bool, default=True, help="consider message dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of node dropout")
    parser.add_argument("--batch_test_flag", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--channel", type=int, default=64, help="hidden channels for model")
    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument('--Ks', nargs='?', default='[5, 10, 20]', help='Output sizes of every layer')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    parser.add_argument("--n_factors", type=int, default=4, help="number of latent factor for user favour")
    parser.add_argument("--ind", type=str, default='distance', help="Independence modeling: mi, distance, cosine")
    parser.add_argument("--kg_file", type=str,default="kg_final")
    # ===== relation context ===== #
    parser.add_argument('--context_hops', type=int, default=3, help='number of context hops')
    parser.add_argument("--num_sample_user2ent", type=int, default=100, help="number of users for each entity")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--agg_n2e", type=str, default="tail")
    parser.add_argument("--score", default="scalar", type=str)
    # ===== save model ===== #
    parser.add_argument("--save", type=bool, default=True, help="save model or not")
    parser.add_argument("--out_dir", type=str, default="./weights/", help="output directory for model")

    return parser.parse_args()
