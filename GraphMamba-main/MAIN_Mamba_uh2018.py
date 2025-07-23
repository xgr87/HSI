import random

import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from Graphormer import ViT,GCN
from functions import metrics,show_results,train_and_test_data,train_epoch,valid_epoch,output_metric,applyPCA,GET_A2,get_data,normalize,GET_dis,get_edge_A
import numpy as np
import time
import os
from Mamba import VisionMamba

parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', choices=['Indian', 'PaviaU', 'Pavia', 'Salinas', 'KSC', 'Botswana', 'HoustonU', 'Houston'],
                    default='HoustonU', help='dataset to use')
parser.add_argument('--mode', choices=['ViT', 'CAF'], default='CAF', help='mode choice')

parser.add_argument("--num_run", type=int, default=10)
parser.add_argument('--epoches', type=int, default=120, help='epoch number')
parser.add_argument('--patches', type=int, default=9, help='number of patches')#奇数#ip11*11 sa 11*11 hu 7*7
parser.add_argument('--PCA_band', type=int, default=20, help='pca_components')#ip70 sa 70 hu 50
parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')


parser.add_argument('--seed', type=int, default=16, help='number of seed')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--batch_size', type=int, default=128, help='number of batch size')
parser.add_argument('--test_freq', type=int, default=10, help='number of evaluation')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#选择cpu或者GPU

seed_value=args.seed
np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)

torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

torch.backends.cudnn.deterministic = True


# -------------------------------------------------------------------------------
# 定位训练和测试样本
# Parameter Setting
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False
# prepare data

input, num_classes, total_pos_train, total_pos_test, total_pos_true, y_train, y_test, y_true = get_data(args.dataset)
##########得到原始图像 训练测试以及所有点坐标 每一类训练测试的个数############
################################################################################################
# normalize data by band norm
input = applyPCA(input, numComponents=args.PCA_band)
################################################################################################
input_normalize = normalize(input)
height, width, band = input_normalize.shape  # 145*145*200
print("height={0},width={1},band={2}".format(height, width, band))
input_normalize = torch.from_numpy(input_normalize.astype(np.float32)).to(device)
# -------------------------------------------------------------------------------
# obtain train and test data
x_train_band, x_test_band, x_true_band, corner_train, corner_test, corner_true, center_pos_train,center_pos_test,center_pos_ture = train_and_test_data(
    input_normalize, band, total_pos_train, total_pos_test, total_pos_true, patch=args.patches, w=height, h=width)
##########得到训练测试以及所有点的光谱############


A_train, dgree_train,_,D2 = GET_A2(x_train_band, input_normalize, corner=corner_train,patches=args.patches , l=3,sigma=10)
dis =GET_dis(args.patches,l=5)
edge = get_edge_A(args.patches)

y_train = torch.from_numpy(y_train).type(torch.LongTensor)  # [695]
Label_train = Data.TensorDataset(x_train_band, y_train, center_pos_train,A_train,dgree_train)

A_test, dgree_test,D_max,D2= GET_A2(x_test_band, input_normalize, corner=corner_test, patches=args.patches ,l=3, sigma=10)

y_test = torch.from_numpy(y_test).type(torch.LongTensor)  # [9671]
Label_test = Data.TensorDataset( x_test_band, y_test, center_pos_test,A_test,dgree_test)



label_train_loader = Data.DataLoader(Label_train, batch_size=args.batch_size, shuffle=True)
##########训练集的光谱值及标签##########
label_test_loader = Data.DataLoader(Label_test, batch_size=args.batch_size, shuffle=True)
# -------------------------------------------------------------------------------

results = []
best_AA2 = []
for run in range(args.num_run):
    best_OA2 = 0.0
    best_AA_mean2 = 0.0
    best_Kappa2 = 0.0
    gcn_net = GCN(height, width, band, num_classes)
    gcn_net = gcn_net.cuda()
    # criterion
    criterion = nn.CrossEntropyLoss().cuda()
    # optimizer
    optimizer = torch.optim.Adam(gcn_net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches // 10, gamma=args.gamma)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epoches*0.5,args.epoches*0.75,args.epoches*0.9],gamma=0.2)  # learning rate decay
    # -------------------------------------------------------------------------------

    tr_net = VisionMamba(
        img_size=args.patches,
        depth=3,
        embed_dim=64,
        channels=band,
        num_classes=num_classes,
        rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        final_pool_type='all', if_abs_pos_embed=True, if_rope=False, if_rope_residual=True, bimamba_type="v2")
    tr_net = tr_net.cuda()
    # optimizer
    optimizer2 = torch.optim.Adam(tr_net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=args.epoches//2, gamma=args.gamma)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=args.epoches // 10, gamma=args.gamma)

    print("start training")
    tic = time.time()
    for epoch in range(args.epoches):
        scheduler.step()
        scheduler2.step()
        # train model
        gcn_net.train()
        tr_net.train()
        train_acc, train_obj, tar_t, pre_t = train_epoch(tr_net, label_train_loader, criterion,
                                                         optimizer2)
        OA1, AA_mean1, Kappa1, AA1 = output_metric(tar_t, pre_t)
        if (epoch % args.test_freq == 0) | (epoch == args.epoches - 1):
            print("Epoch: {:03d} train_loss: {:.4f} train_acc: {:.4f}".format(epoch + 1, train_obj, train_acc))

        if (epoch % args.test_freq == 0) | (epoch == args.epoches - 1)and epoch>=args.epoches*0.8:

            gcn_net.eval()
            tr_net.eval()
            tar_v, pre_v = valid_epoch( tr_net, label_test_loader, criterion)
            OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)
            if OA2 >= best_OA2 :
                best_OA2 = OA2
                best_AA_mean2 = AA_mean2
                best_Kappa2 = Kappa2
                best_AA2 = AA2
    #             run_results = metrics(best_OA2, best_AA_mean2, best_Kappa2,AA2)
    # show_results(
    #     run_results,agregated=False)
    # results.append(run_results)
    toc = time.time()

    f = open('./result/' + args.dataset + '_results.txt', 'a+')

    str_results = '\n\n************************************************' \
                  + '\nseed_value={}'.format(seed_value) \
                  + '\nepoch={}'.format(epoch) \
                  + '\nPCA_band={}'.format(args.PCA_band) \
                  + '\nOA={:.2f}'.format(best_OA2*100) \
                  + '\nAA={:.2f}'.format(best_AA_mean2*100) \
                  + '\nKappa={:.2f}'.format(best_Kappa2*100) \
                  + '\nbest_AA2=' + str(np.around(best_AA2*100, 2))


    f.write(str_results)
    f.close()

    print('\nbest_OA2={}'.format(best_OA2))
    print('\nbest_AA_mean2={}'.format(best_AA_mean2))
    print('\nbest_Kappa2={}'.format(best_Kappa2))

if args.num_run > 1:
    show_results(results,  agregated=True)









