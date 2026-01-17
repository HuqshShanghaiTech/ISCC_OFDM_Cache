import matplotlib
from matplotlib import ticker

import matplotlib.pyplot as plt
import copy
import numpy as np
import torch
import random
import logging
import time
import os
import csv

from cvx_journal import *
from torchvision import datasets, transforms
from utils.sampling import sensing_data_dict
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import *
from models.Fed import FedAvg
from models.test import test_img
from data_set import *
from Par import *
# from main_fed import inverse_local_ep
from opt_power_adp_batch import cal_batch_round_increasing, cal_batch_round_decreasing

args = args_parser()
log_name = 'Fed-{}-{}-m7-user-{}-round-BS-{}-{}-{}-{}.log'.format(time.strftime('%Y-%m-%dT%H-%M-%S'), args.num_users,
                                                                  args.epochs, args.rule, args.eta, args.T, args.E)
# 创建一个logger
logger = logging.getLogger('CNN')
logger.setLevel(logging.DEBUG)

# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# 定义handler的输出格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

# 给logger添加handler
logger.addHandler(ch)

# logger.info('fed_learn_mnist_cnn_100_iid_v2')
logger.info('fed_isac_cnn')
logger.info('OFDM')

figure_save_path = "figure"
if not os.path.exists(figure_save_path):
    os.makedirs(figure_save_path)
"""
def compute_expression(
    w_local,
    w_global
):
    # 构造 g_tilde
    g_list = []
    for key in w_global.keys():
        diff = w_local[key] - w_global[key]
        g_list.append(diff.view(-1))
    g_tilde = torch.cat(g_list)

    # N = ||g||_2^2
    N = torch.sum(g_tilde ** 2)

    # x_max, x_min
    g_abs = torch.abs(g_tilde)
    x_max = torch.max(g_abs)
    x_min = torch.min(g_abs)

    # i = 梯度维度
    i = g_tilde.numel()

    return N, x_max-x_min
"""
if __name__ == '__main__':
    start_time = time.time()
    # parse args
    args = args_parser()
    # args, unparsed = args_parser.parse_known_args()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.7723, 0.8303, 0.9284), (0.3916, 0.3057, 0.1893)),
    ])


    net_glob = ResNet.ResNet10().to(args.device)
    # net_glob = CNNCifar(args).to(args.device)

    model_root = './save/models/models_10_m7.pth'
    if os.path.exists(model_root) is False:
        torch.save(net_glob.state_dict(), model_root)
    net_glob.load_state_dict(torch.load(model_root))

    net_total_params = sum(p.numel() for p in net_glob.parameters())
    print('| net_total_params: {}'.format(net_total_params))
    print(net_glob)

    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    logger.info(args)
    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]

    batch_size_init = args.local_bs

    loss_train_init = 6

    root_radar_1 = './data/spect/THREE_RADAR_3000/radar_1/'
    root_radar_2 = './data/spect/THREE_RADAR_3000/radar_2/'
    root_radar_3 = './data/spect/THREE_RADAR_3000/radar_3/'

    num_hat = [0]*Devnum
    beta_1 = [0]*Devnum
    p_s_late = []
    b_k_late = []
    loss_plt = []
    acc_plt = []
    diff = 0
    N = 0

    for tmp_round in range(1, args.epochs + 1):

        if tmp_round == 1:
            num, p_s, alpha = solve_optimization_problem(tmp_round, p_s_late, b_k_late, N, diff, None, None, None)
            alpha_1 = alpha

        else:
            num, num_hat, p_s, alpha, beta= solve_optimization_problem(tmp_round, p_s_late, num_hat, N, diff, num, alpha_1, beta_1)
            alpha_1 = alpha
            beta_1 = beta
        #num = [int(x // 7) for x in [273., 273., 273., 273., 273., 273.]]
        num_hat = [5] * Devnum
        print(num)
        print(num_hat)
        dataset_train_1 = MyDatasetrain_1(txt=root_radar_1 + 'train_1_m7.txt', transform=data_transform, num = num[0], num_hat = num_hat[0], tmp_round = tmp_round)
        # dataset_test_1 = MyDataset(txt=root_radar_1 + 'test_1.txt', transform=data_transform)

        dataset_train_2 = MyDatasetrain_2(txt=root_radar_2 + 'train_1_m7.txt', transform=data_transform, num = num[1], num_hat = num_hat[1], tmp_round =tmp_round)
        # dataset_test_2 = MyDataset(txt=root_radar_2 + 'test_1.txt', transform=data_transform)

        dataset_train_3 = MyDatasetrain_3(txt=root_radar_3 + 'train_1_m7.txt', transform=data_transform, num = num[2], num_hat = num_hat[2], tmp_round =tmp_round)
        # dataset_test_3 = MyDataset(txt=root_radar_3 + 'test_1.txt', transform=data_transform)

        dataset_train_4 = MyDatasetrain_4(txt=root_radar_1 + 'train_2_m7.txt', transform=data_transform, num = num[3], num_hat = num_hat[3], tmp_round =tmp_round)
        # dataset_test_4 = MyDataset(txt=root_radar_1 + 'test_2.txt', transform=data_transform)

        dataset_train_5 = MyDatasetrain_5(txt=root_radar_2 + 'train_2_m7.txt', transform=data_transform, num = num[4], num_hat = num_hat[4], tmp_round =tmp_round)
        # dataset_test_5 = MyDataset(txt=root_radar_2 + 'test_2.txt', transform=data_transform)

        dataset_train_6 = MyDatasetrain_6(txt=root_radar_3 + 'train_2_m7.txt', transform=data_transform, num = num[5], num_hat = num_hat[5], tmp_round =tmp_round)
        # dataset_test_6 = MyDataset(txt=root_radar_3 + 'test_3.txt', transform=data_transform)

        dataset_test = MyDataset(txt='./data/spect/THREE_RADAR_3000/' + 'test_m7.txt', transform=data_transform)
        dataset_train = [dataset_train_1, dataset_train_2, dataset_train_3, dataset_train_4, dataset_train_5, dataset_train_6]
        dict_users = sensing_data_dict(dataset_train_1, dataset_train_2, dataset_train_3, dataset_train_4, dataset_train_5, dataset_train_6)

        local_steps = args.local_ep

        #p_s = [0.02999983, 0.02999983, 0.02999983, 0.02999983, 0.02999983, 0.02999983]

        #img_size = dataset_train_1[0][0].shape
        #print(img_size)

        # shuffle the data index in order to fetch data
        # for idx_radar in range(3):
        #     rand_index_ = list(range(12000))
        #     random.shuffle(rand_index_)



        # batch data into new dataset
        batch_size = 128

        loss_locals = []

        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        for idx in range(Devnum):
            local = LocalUpdate(args=args, batch_size=batch_size, dataset=dataset_train[idx], idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device), local_steps=local_steps, p_s=p_s[idx])
            #N, diff = compute_expression(
            #    w_local=w,
            #    w_global=w_glob,
            #)
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)

        """
        #AirComp 噪声
        n_var = 1
        #eta = 0.001
        #eta = args.eta
        # eta = random.uniform(0, 0.02)
        aircomp_noise = np.random.normal(0, n_var / (eta[0] ** 0.5))
        aircomp_noise = 0

        for key in w_glob.keys():
            w_glob[key] = w_glob[key] - aircomp_noise * 0.1
"""
        #weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)

        # testing
        net_glob.eval()
        acc_test_1, loss_train_1 = test_img(net_glob, dataset_test, args)

        #绘图信息
        loss_plt.append(loss_avg)
        acc_plt.append(acc_test_1.item() / 100)

        #控制台输出信息
        logger.info('Epoch: {}'.format(tmp_round))
        #logger.info('eta: {}'.format(eta[0]))
        #logger.info('AirComp noise: {}'.format(aircomp_noise))
        logger.info('Train loss: {:.4f}'.format(loss_avg))
        logger.info("average test acc: {:.2f}%".format(acc_test_1))
        # print("Test on dataset from Radar 2: {:.3f}%, training loss: {:.6f}".format(acc_test_2, loss_train_2))

        p_s_late = p_s
        b_k_late = num
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"程序执行时间为：{elapsed_time:.6f} 秒")
    acc_save_path = "acc/diff para"
    acc_save_path_1 = "/sdb/hqs/ISCC/acc/diff para"
    if not os.path.exists(acc_save_path_1):
        os.makedirs(acc_save_path_1)
    if not os.path.exists(acc_save_path):
        os.makedirs(acc_save_path)
    with open(os.path.join(acc_save_path_1, 'acc_ofdm_cache_punish_20.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Round', 'Accuracy'])
        for i, acc in enumerate(acc_plt):
            writer.writerow([i + 1, acc])
    with open(os.path.join(acc_save_path, 'acc_ofdm_cache_punish_20.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Round', 'Accuracy'])
        for i, acc in enumerate(acc_plt):
            writer.writerow([i + 1, acc])
    
    figure_save_path = "figure"
    figure_save_path_1 = "/sdb/hqs/ISCC/figure"
    if not os.path.exists(figure_save_path_1):
        os.makedirs(figure_save_path_1)
    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path)

    

    end_time = time.time()
    loss_num = [i for i in range(1, len(loss_plt) + 1)]
    plt.figure()
    plt.plot(loss_num, loss_plt)

    plt.legend(['loss'])
    plt.savefig(os.path.join(figure_save_path, 'loss_ofdm_cache_punish_20.png'))
    plt.xlabel('Global Communication Round')
    plt.ylabel('Loss')
    plt.show()

    plt.figure()
    plt.plot(loss_num, acc_plt)
    plt.legend(['acc'])
    plt.savefig(os.path.join(figure_save_path, 'acc_ofdm_cache_punish_20.png'))
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    plt.xlabel('Global Communication Round')
    plt.ylabel('Testing Accuracy')
    plt.show()

    print('CVX solve 2025.7.8 17:16 ofdm_cache_punish_20')