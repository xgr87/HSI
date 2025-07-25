from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import numpy as np
import scipy.io as sio
import torch
import math
from sklearn import preprocessing
import h5py


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#选择cpu或者GPU

def chooose_train_and_test_point(train_data, test_data, true_data, num_classes):
    number_train = []
    pos_train = {}
    number_test = []
    pos_test = {}
    number_true = []
    pos_true = {}
    #-------------------------for train data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(train_data==(i+1))
        number_train.append(each_class.shape[0])
        pos_train[i] = each_class

    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        total_pos_train = np.r_[total_pos_train, pos_train[i]] #(695,2)
    total_pos_train = total_pos_train.astype(int)
    #--------------------------for test data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(test_data==(i+1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class

    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]] #(9671,2)
    total_pos_test = total_pos_test.astype(int)
    #--------------------------for true data------------------------------------
    for i in range(num_classes+1):
        each_class = []
        each_class = np.argwhere(true_data==i)
        number_true.append(each_class.shape[0])
        pos_true[i] = each_class

    total_pos_true = pos_true[0]
    for i in range(1, num_classes+1):
        total_pos_true = np.r_[total_pos_true, pos_true[i]]
    total_pos_true = total_pos_true.astype(int)

    return total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true
#-------------------------------------------------------------------------------
# 边界拓展：镜像
def mirror_hsi(height,width,band,input_normalize,patch=5):
    padding=patch//2
    mirror_hsi=np.zeros((height+2*padding,width+2*padding,band),dtype=float)
    #中心区域
    mirror_hsi[padding:(padding+height),padding:(padding+width),:]=input_normalize
    #左边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),i,:]=input_normalize[:,padding-i-1,:]
    #右边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:]=input_normalize[:,width-1-i,:]
    #上边镜像
    for i in range(padding):
        mirror_hsi[i,:,:]=mirror_hsi[padding*2-i-1,:,:]
    #下边镜像
    for i in range(padding):
        mirror_hsi[height+padding+i,:,:]=mirror_hsi[height+padding-1-i,:,:]

    print("**************************************************")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0],mirror_hsi.shape[1],mirror_hsi.shape[2]))
    print("**************************************************")
    return mirror_hsi
#-------------------------------------------------------------------------------
# 排序取索引
def choose_top(image,cornor_index,x,y,patch,b,n_top):
    sort = image.reshape(patch * patch, b)
    sort = torch.from_numpy(sort).type(torch.FloatTensor)
    pos = (x - cornor_index[0]) * patch + (y - cornor_index[1])
    Q = torch.sum(torch.pow(sort[pos] - sort, 2), dim=1)
    _, indices = Q.topk(k=n_top, dim=0, largest=False, sorted=True)
    return indices
#-------------------------------------------------------------------------------
# 获取patch的图像数据
def gain_neighborhood_pixel(pca_image, point, i, patch, W, H):
    x = point[i,0]
    y = point[i,1]
    m=int((patch-1)/2)##patch奇数
    _,_,b=pca_image.shape
    if x<=m:
        if y<=m:
            temp_image = pca_image[0:patch, 0:patch, :]
            cornor_index = [0,0]
        if y>=(H-m):
            temp_image = pca_image[0:patch, H-patch:H, :]
            cornor_index = [0, H-patch]
        if y>m and y<H-m:
            temp_image = pca_image[0:patch, y-m:y+m+1, :]
            cornor_index = [0, y-m]
    if x>=(W-m):
        if y<=m:
            temp_image = pca_image[W-patch:W, 0:patch, :]
            cornor_index = [W-patch, 0]
        if y>=(H-m):
            temp_image = pca_image[W-patch:W, H-patch:H, :]
            cornor_index = [W - patch, H-patch]
        if y>m and y<H-m:
            temp_image = pca_image[W-patch:W, y-m:y+m+1, :]
            cornor_index = [W - patch, y-m]
    if x>m and x<W-m:
        if y<=m:
            temp_image = pca_image[x-m:x+m+1, 0:patch, :]
            cornor_index = [x-m, 0]
        if y>=(H-m):
            temp_image = pca_image[x-m:x+m+1, H-patch:H, :]
            cornor_index = [x - m, H-patch]
        if y>m and y<H-m:
            temp_image = pca_image[x-m:x+m+1, y-m:y+m+1, :]
            cornor_index = [x - m, y-m]
            # look11=pca_image[:,:,0]
            # look12=temp_image[:,:,0]
            # print(temp_image.shape)
    center_pos = (x - cornor_index[0]) * patch + (y - cornor_index[1])
    return temp_image,cornor_index,center_pos
# 汇总训练数据和测试数据
def train_and_test_data(pca_image, band, train_point, test_point, true_point, patch, w, h):
    x_train = torch.zeros((train_point.shape[0], patch, patch, band), dtype=torch.float32).to(device)
    x_test = torch.zeros((test_point.shape[0], patch, patch, band), dtype=torch.float32).to(device)
    x_true = torch.zeros((true_point.shape[0], patch, patch, band), dtype=torch.float32).to(device)
    corner_train = np.zeros((train_point.shape[0], 2), dtype=int)
    corner_test = np.zeros((test_point.shape[0], 2), dtype=int)
    corner_true = np.zeros((true_point.shape[0], 2), dtype=int)
    center_pos_train = torch.zeros((train_point.shape[0]), dtype=int).to(device)
    center_pos_test = torch.zeros((test_point.shape[0]), dtype=int).to(device)
    center_pos_ture = torch.zeros((true_point.shape[0]), dtype=int).to(device)
    for i in range(train_point.shape[0]):
        x_train[i,:,:,:],corner_train[i,:],center_pos_train[i]= gain_neighborhood_pixel(pca_image, train_point, i, patch, w, h)
    for j in range(test_point.shape[0]):
        x_test[j,:,:,:],corner_test[j,:],center_pos_test[j] = gain_neighborhood_pixel(pca_image, test_point, j, patch, w, h)
    for k in range(true_point.shape[0]):
        x_true[k,:,:,:],corner_true[k,:],center_pos_ture[k] = gain_neighborhood_pixel(pca_image, true_point, k, patch, w, h)
    print("x_train shape = {}, type = {}".format(x_train.shape,x_train.dtype))
    print("x_test  shape = {}, type = {}".format(x_test.shape,x_test.dtype))
    print("x_true  shape = {}, type = {}".format(x_true.shape,x_test.dtype))
    print("**************************************************")

    return x_train.reshape(train_point.shape[0], patch*patch,band), x_test.reshape(test_point.shape[0], patch*patch,band), x_true.reshape(true_point.shape[0], patch*patch,band),corner_train,corner_test,corner_true,center_pos_train,center_pos_test,center_pos_ture
#-------------------------------------------------------------------------------
# 标签y_train, y_test
def train_and_test_label(number_train, number_test, number_true, num_classes):
    y_train = []
    y_test = []
    y_true = []
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)
    for i in range(num_classes):
        for j in range(number_true[i]):
            y_true.append(i)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_true = np.array(y_true)
    print("y_train: shape = {} ,type = {}".format(y_train.shape,y_train.dtype))
    print("y_test: shape = {} ,type = {}".format(y_test.shape,y_test.dtype))
    print("y_true: shape = {} ,type = {}".format(y_true.shape,y_true.dtype))
    print("**************************************************")
    return y_train, y_test, y_true
#-------------------------------------------------------------------------------
class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt
#-------------------------------------------------------------------------------
def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res, target, pred.squeeze()
#-------------------------------------------------------------------------------
# train model
def train_epoch(tr_net, train_loader, criterion,optimizer2):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target, center_pos,A,D) in enumerate(train_loader):
        batch_A = A.cuda()
        D=D.cuda()
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()

        optimizer2.zero_grad()
        batch_pred = tr_net(batch_data,center_pos,batch_A)
        loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer2.step()

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return top1.avg, objs.avg, tar, pre
#-------------------------------------------------------------------------------
# validate model
def valid_epoch(tr_net, valid_loader, criterion):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target,center_pos,A,D) in enumerate(valid_loader):
        batch_A = A.cuda()
        D=D.cuda()
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()

        batch_pred = tr_net(batch_data,center_pos,batch_A)
        loss = criterion(batch_pred, batch_target)

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
        
    return tar, pre

def test_epoch(model, test_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(test_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()   

        batch_pred = model(batch_data)

        _, pred = batch_pred.topk(1, 1, True, True)
        pp = pred.squeeze()
        pre = np.append(pre, pp.data.cpu().numpy())
    return pre
#-------------------------------------------------------------------------------
def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA
#-------------------------------------------------------------------------------
def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float64)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA

def applyPCA(X, numComponents):

    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

    return newX


def GET_A(temp_image,D,corner,l,sigma=10, w_all=145, h_all=145):#l为邻域范围，sigma为计算距离的参数
    N,h,w,_=temp_image.shape
    B = np.zeros((w * h, w * h), dtype=np.float32)
    for i in range(h):  # 图像的行  h代表有几行，w代表有几列
        for j in range(w):  # 图像的列
            m = int(i * w + j)  # 在邻接矩阵中的行数
            for k in range(l):  # 邻域的行数
                for q in range(l):  # 邻域的列数
                    n = int((i + (k - (l - 1) / 2)) * w + (j + (q - (l - 1) / 2)))  # 计算邻域，并转换为邻域在邻接矩阵中的列数
                    if 0 <= i + (k - (l - 1) / 2) < h and 0 <= (j + (q - (l - 1) / 2)) < w and m != n:
                        B[m, n] = 1
##############得到每个输入图像的7*7图像的邻域矩阵###############################
    index=np.argwhere(B == 1)#得到邻域矩阵中不为零的索引
    index_num,_=index.shape
    X = np.zeros((index_num,2),dtype=np.int64)
    Y = np.zeros((index_num, 2), dtype=np.int64)
    for i in range(index_num):
        X[i, 0] = index[i, 0] // w #邻域矩阵行值在图像中行坐标
        X[i, 1] = index[i, 0] % w#邻域矩阵行值在图像中列坐标
        Y[i, 0] = index[i, 1] // w#邻域矩阵列值在图像中行坐标
        Y[i, 1] = index[i, 1] % w#邻域矩阵列值在图像中列坐标
##############得到每个输入图像的7*7图像的二维坐标###############################
    X_N = np.zeros((N,index_num,2), dtype=np.int64)#在原始图像上的行值
    Y_N = np.zeros((N, index_num, 2), dtype=np.int64)#在原始图像上的列值
    corner_N = np.expand_dims(corner, 1).repeat(index_num, axis=1)
    X=np.expand_dims(X, 0).repeat(N, axis=0)
    Y = np.expand_dims(Y, 0).repeat(N, axis=0)
    X_N[:, :, 0] = X[:,:,0] + corner_N[:,:,0]#在原始图像上的行值
    X_N[:, :, 1] = X[:,:,1] + corner_N[:,:,1]#在原始图像上的列值
    X_A=X_N[:, :, 0]*w_all+X_N[:, :, 1]#在原始图像邻接矩阵距离索引
    Y_N[:, :, 0] = Y[:,:,0] + corner_N[:,:,0]#在原始图像上的行值
    Y_N[:, :, 1] = Y[:,:,1] + corner_N[:,:,1]#在原始图像上的列值
    Y_A = Y_N[:, :, 0] * w_all + Y_N[:, :, 1]#在原始图像邻接矩阵距离索引

    A = np.zeros((N, w * h, w * h), dtype=np.float32)
    A = torch.from_numpy(A).type(torch.FloatTensor).cuda()
    index2 = np.where(B == 1)  # 得到邻域矩阵中不为零的索引
    for i in range(N):
        C=torch.from_numpy(B).type(torch.FloatTensor).cuda()
        C[index2[0],index2[1]]= D[X_A[i], Y_A[i]]
        A[i,:,:] = C

    return A

def GET_A2(temp_image,input2,corner,patches,l,sigma=10,):#l为邻域范围，sigma为计算距离的参数
    input2=input2.cuda()
    N,_,_=temp_image.shape
    w = patches
    h = patches
    B = torch.zeros((w * h, w * h), dtype=torch.float32).to(device)
    for i in range(h):  # 图像的行  h代表有几行，w代表有几列
        for j in range(w):  # 图像的列
            m = int(i * w + j)  # 在邻接矩阵中的行数
            for k in range(l):  # 邻域的行数
                for q in range(l):  # 邻域的列数
                    n = int((i + (k - (l - 1) / 2)) * w + (j + (q - (l - 1) / 2)))  # 计算邻域，并转换为邻域在邻接矩阵中的列数
                    if 0 <= i + (k - (l - 1) / 2) < h and 0 <= (j + (q - (l - 1) / 2)) < w and m != n:
                        B[m, n] = 1
##############得到每个输入图像的7*7图像的邻域矩阵###############################
    index2 = torch.where(B == 1)  # 得到邻域矩阵中不为零的索引
    A = torch.zeros((N, w * h, w * h), dtype=torch.float32).to(device)

    for i in range(N):#####corenor为左上角的值
        C = torch.Tensor(B)
        x_l=int(corner[i,0])
        x_r=int(corner[i,0]+patches)
        y_l=int(corner[i,1])
        y_r=int(corner[i,1]+patches)
        D = pdists_corner(input2[x_l:x_r,y_l:y_r,:],sigma)
        # D = D.cpu().numpy()
        C[index2[0], index2[1]] = D[index2[0],index2[1]]
        A[i,:,:] = C
    D = A.sum(2).round().type(torch.int)
    D_max=torch.max(D).type(torch.int)
    # m, dgree_train = D.sort(dim=1)  # 由小到大排序
    # m2, dgree_train = dgree_train.sort(dim=1)  # 对应位置的大小排名 排名越小值越小
    D2 =B.sum(1).round().type(torch.int).cuda()
    return A, D, D_max,D2


def pdists_corner(A,sigma=10):
    height,width, band = A.shape
    A=A.reshape(height * width, band)
    prod = torch.mm(A, A.t())#21025*21025
    norm = prod.diag().unsqueeze(1).expand_as(prod)#21025*21025
    res = (norm + norm.t() - 2 * prod).clamp(min=0)
    D =torch.exp(-res/(sigma ** 2))
    return D

def pdists(A,sigma=10):
    A=A.cuda()
    prod = torch.mm(A, A.t())#21025*21025
    norm = prod.diag().unsqueeze(1).expand_as(prod)#21025*21025
    res = (norm + norm.t() - 2 * prod).clamp(min=0)
    D =torch.exp(-res/(sigma ** 2))
    return D

def normalize(data):
    height, width, bands = data.shape
    data = np.reshape(data, [height * width, bands])
    minMax = preprocessing.StandardScaler()
    data = minMax.fit_transform(data)#计算训练数据的均值和方差，还会基于计算出来的均值和方差来转换训练数据，从而把数据转换成标准的正太分布
    data = np.reshape(data, [height, width, bands])
    return data

################get data######################################################################################################################
def load_dataset(Dataset):
    if Dataset == 'Indian':
        mat_data = sio.loadmat('Datasets/Indian_pines.mat')
        mat_gt = sio.loadmat('Datasets/Indian_pines_gt.mat')
        data_hsi = mat_data['indian_pines']
        # data_hsi = data_hsi[:,:,::9]
        gt_hsi = mat_gt['indian_pines_gt']
        TOTAL_SIZE = 10249
        VALIDATION_SPLIT = 0.97
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'PaviaU':
        uPavia = sio.loadmat('Datasets/PaviaU.mat')
        gt_uPavia = sio.loadmat('Datasets/PaviaU_gt.mat')
        data_hsi = uPavia['paviaU']
        data_hsi = data_hsi[:, :, ::5]
        gt_hsi = gt_uPavia['paviaU_gt']
        TOTAL_SIZE = 42776
        VALIDATION_SPLIT = 0.995
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'Pavia':
        uPavia = sio.loadmat('/home/yat/datasets/Pavia.mat')
        gt_uPavia = sio.loadmat('/home/yat/datasets/Pavia_gt.mat')
        data_hsi = uPavia['pavia']
        gt_hsi = gt_uPavia['pavia_gt']
        TOTAL_SIZE = 148152
        VALIDATION_SPLIT = 0.999
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'Salinas':
        SV = sio.loadmat('Datasets/Salinas.mat')
        gt_SV = sio.loadmat('Datasets/Salinas_gt.mat')
        data_hsi = SV['salinas']
        data_hsi = data_hsi[:, :, ::2]
        gt_hsi = gt_SV['salinas_gt']
        TOTAL_SIZE = 54129
        VALIDATION_SPLIT = 0.995
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'KSC':
        KSC = sio.loadmat('/home/yat/datasets/KSC.mat')
        gt_KSC = sio.loadmat('/home/yat/datasets/KSC_gt.mat')
        data_hsi = KSC['KSC']
        gt_hsi = gt_KSC['KSC_gt']
        TOTAL_SIZE = 5211
        VALIDATION_SPLIT = 0.95
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'Botswana':
        BS = sio.loadmat('/home/yat/datasets/Botswana.mat')
        gt_BS = sio.loadmat('/home/yat/datasets/Botswana_gt.mat')
        data_hsi = BS['Botswana']
        gt_hsi = gt_BS['Botswana_gt']
        TOTAL_SIZE = 3248
        VALIDATION_SPLIT = 0.99
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)
    if Dataset == 'Botswana':
        BS = sio.loadmat('/home/yat/datasets/Botswana.mat')
        gt_BS = sio.loadmat('/home/yat/datasets/Botswana_gt.mat')
        data_hsi = BS['Botswana']
        gt_hsi = gt_BS['Botswana_gt']
        TOTAL_SIZE = 3248
        VALIDATION_SPLIT = 0.99
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)
    if Dataset == 'Houston':
        BS = sio.loadmat('/home/yat/datasets/Houston.mat')
        gt_BS = sio.loadmat('/home/yat/datasets/Houston_gt.mat')
        data_hsi = BS['Houston']
        gt_hsi = gt_BS['Houston_gt']
        TOTAL_SIZE = 664845
        VALIDATION_SPLIT = 0.05
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)
    if Dataset == 'HoustonU':
        BS = h5py.File('/home/yat/datasets/HoustonU.mat','r')
        gt_BS = h5py.File('/home/yat/datasets/HoustonU_gt.mat')
        print(BS.keys())
        print(gt_BS.keys())
        # data_hsi = np.transpose(BS['ori_data'])
        # gt_hsi = np.transpose(gt_BS['map'])

        # BS = sio.loadmat('/home/yat/datasets/HoustonU.mat')
        # gt_BS = sio.loadmat('/home/yat/datasets/HoustonU_gt.mat')
        data_hsi = np.transpose(BS['houstonU'])
        gt_hsi = np.transpose(gt_BS['houstonU_gt'])
        TOTAL_SIZE = 119200
        VALIDATION_SPLIT = 0.05
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)
    return data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE, VALIDATION_SPLIT


def sampling(proportion, ground_truth,CLASSES_NUM,dataset=None):
    if dataset == 'Indian':
        proportion=0.1
    elif dataset == 'PaviaU':
        proportion = 0.1
    elif dataset == 'Salinas':
        proportion = 0.1
    elif dataset == 'HoustonU':
        proportion = 0.003
    else:
        proportion =1

    train = {}
    test = {}
    train_num = []
    test_num = []
    labels_loc = {}
    for i in range(CLASSES_NUM):
        indexes = np.argwhere(ground_truth == (i + 1))
        np.random.shuffle(indexes)#打乱顺序
        labels_loc[i] = indexes
        if proportion != 1:
            nb_val = max(int((proportion) * len(indexes)), 3)
            # if indexes.shape[0]<=60:
            #     nb_val = 15
            # else:
            #     nb_val = 30
        else:
            nb_val = 0
        # print(i, nb_val, indexes[:nb_val])
        # train[i] = indexes[:-nb_val]
        # test[i] = indexes[-nb_val:]
        train_num.append(nb_val)
        test_num.append(len(indexes)-nb_val)
        train[i] = indexes[:nb_val]
        test[i] = indexes[nb_val:]
    train_indexes =train[0]
    test_indexes = test[0]
    for i in range(CLASSES_NUM-1):
        train_indexes= np.concatenate((train_indexes,train[i+1]),axis=0)
        test_indexes= np.concatenate((test_indexes,test[i+1]),axis=0)
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes,train_num,test_num#返回训练集和测试集的索引


def index_change(index,w):
    N=len(index)
    index2=np.zeros((N,2),dtype=int)
    for i in range(N):
        index2[i, 0] = index[i] // w
        index2[i, 1] = index[i] % w
    return index2
def get_label(indices,gt_hsi):
    dim_0 = indices[:, 0]
    dim_1 = indices[:, 1]
    label=gt_hsi[dim_0,dim_1]
    return label


def get_data(dataset):
    data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE,VALIDATION_SPLIT = load_dataset(dataset)
    gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]),)
    CLASSES_NUM = max(gt)
    train_indices, test_indices,train_num,test_num = sampling(VALIDATION_SPLIT, gt_hsi, CLASSES_NUM,dataset)
    _, total_indices,_,total_num = sampling(1, gt_hsi , CLASSES_NUM)
    y_train = get_label(train_indices, gt_hsi)-1
    y_test = get_label(test_indices, gt_hsi)-1
    y_true = get_label(total_indices, gt_hsi)-1
    return  data_hsi,CLASSES_NUM,train_indices,test_indices,total_indices,y_train, y_test, y_true

def metrics(best_OA2, best_AA_mean2, best_Kappa2,AA2):
    results = {}
    results["OA"] = best_OA2 * 100.0
    results['AA'] = best_AA_mean2 * 100.0
    results["Kappa"] = best_Kappa2 * 100.0
    results["class acc"] = AA2 * 100.0
    return results
def show_results(results, agregated=False):
    text = ""

    if agregated:
        accuracies = [r["OA"] for r in results]
        aa = [r['AA'] for r in results]
        kappas = [r["Kappa"] for r in results]
        class_acc = [r["class acc"] for r in results]

        class_acc_mean = np.mean(class_acc, axis=0)
        class_acc_std = np.std(class_acc, axis=0)

    else:
        accuracy = results["OA"]
        aa = results['AA']
        classacc = results["class acc"]
        kappa = results["Kappa"]

    text += "---\n"
    text += "class acc :\n"
    if agregated:
        for score, std in zip(class_acc_mean,
                                     class_acc_std):
            text += "\t{:.02f} +- {:.02f}\n".format(score, std)
    else:
        for score in classacc:
            text += "\t {:.02f}\n".format(score)
    text += "---\n"

    if agregated:
        text += ("OA: {:.02f} +- {:.02f}\n".format(np.mean(accuracies),
                                                         np.std(accuracies)))
        text += ("AA: {:.02f} +- {:.02f}\n".format(np.mean(aa),
                                                   np.std(aa)))
        text += ("Kappa: {:.02f} +- {:.02f}\n".format(np.mean(kappas),
                                                      np.std(kappas)))
    else:
        text += "OA : {:.02f}%\n".format(accuracy)
        text += "AA: {:.02f}%\n".format(aa)
        text += "Kappa: {:.02f}\n".format(kappa)

    print(text)

############################A_star 算法##################################
def a_star_search(grid: list, begin_point: list, target_point: list, cost=1):
    assert ((grid[begin_point[0]][begin_point[1]] != 1) and (grid[target_point[0]][target_point[1]] != 1))

    # the cost map which pushes the path closer to the goal
    heuristic = [[0 for row in range(len(grid[0]))] for col in range(len(grid))]
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            heuristic[i][j] = abs(i - target_point[0]) + abs(j - target_point[1])
            if grid[i][j] == 1:
                heuristic[i][j] = 99  # added extra penalty in the heuristic map

    # the actions we can take
    delta = [[-1, 0],  # go up
             [0, -1],  # go left
             [1, 0],  # go down
             [0, 1]]  # go right

    close_matrix = [[0 for col in range(len(grid[0]))] for row in range(len(grid))]  # the referrence grid
    close_matrix[begin_point[0]][begin_point[1]] = 1
    action_matrix = [[0 for col in range(len(grid[0]))] for row in range(len(grid))]  # the action grid

    x = begin_point[0]
    y = begin_point[1]
    g = 0
    f = g + heuristic[begin_point[0]][begin_point[0]]
    cell = [[f, g, x, y]]

    found = False  # flag that is set when search is complete
    resign = False  # flag set if we can't find expand

    while not found and not resign:
        if len(cell) == 0:
            resign = True
            return None, None
        else:
            cell.sort()  # to choose the least costliest action so as to move closer to the goal
            cell.reverse()
            next = cell.pop()
            x = next[2]
            y = next[3]
            g = next[1]
            f = next[0]

            if x == target_point[0] and y == target_point[1]:
                found = True
            else:
                # delta have four steps
                for i in range(len(delta)):  # to try out different valid actions
                    x2 = x + delta[i][0]
                    y2 = y + delta[i][1]
                    if x2 >= 0 and x2 < len(grid) and y2 >= 0 and y2 < len(grid[0]):  # 判断可否通过那个点
                        if close_matrix[x2][y2] == 0 and grid[x2][y2] == 0:
                            g2 = g + cost
                            f2 = g2 + heuristic[x2][y2]
                            cell.append([f2, g2, x2, y2])
                            close_matrix[x2][y2] = 1
                            action_matrix[x2][y2] = i
    invpath = []
    x = target_point[0]
    y = target_point[1]
    invpath.append([x, y])  # we get the reverse path from here
    while x != begin_point[0] or y != begin_point[1]:
        x2 = x - delta[action_matrix[x][y]][0]
        y2 = y - delta[action_matrix[x][y]][1]
        x = x2
        y = y2
        invpath.append([x, y])

    path = []
    for i in range(len(invpath)):
        path.append(invpath[len(invpath) - 1 - i])
    return path, action_matrix

def path_search(grid,begin,target):
    a_star_path, action_matrix = a_star_search(grid, begin, target)
    a_star_path=np.array(a_star_path)
    edge_number=np.zeros(4)
    for i in range(a_star_path.shape[0]-1):
        x= a_star_path[i+1,0] - a_star_path[i,0]
        y = a_star_path[i+1, 1] - a_star_path[i, 1]
        if x==-1 and y==0:
            edge_number[0]=edge_number[0]+1
        if x==0 and y==-1:
            edge_number[1]=edge_number[1]+1
        if x==1 and y==0:
            edge_number[2]=edge_number[2]+1
        if x==0 and y==1:
            edge_number[3]=edge_number[3]+1
    return edge_number

def GET_dis(patches,l):  # l为邻域范围，sigma为计算距离的参数
    dis=torch.zeros((patches*patches,patches*patches),dtype=torch.int64)
    h=patches
    w=patches
    center=(int)(l-1)/2
    for i in range(h):  # 图像的行  h代表有几行，w代表有几列
        for j in range(w):  # 图像的列
            m = int(i * w + j)  # 在邻接矩阵中的行数
            for k in range(l):  # 邻域的行数
                for q in range(l):  # 邻域的列数
                    n = int((i + (k - (l - 1) / 2)) * w + (j + (q - (l - 1) / 2)))  # 计算邻域，并转换为邻域在邻接矩阵中的列数
                    if 0 <= i + (k - (l - 1) / 2) < h and 0 <= (j + (q - (l - 1) / 2)) < w :
                        if abs(k-center)==0 and abs(q-center)==0:
                            dis[m, n] = 1
                        if (abs(k-center)==0 and abs(q-center)==1) or (abs(k-center)==1 and abs(q-center)==0):
                            dis[m, n] = 2
                        if  abs(k-center)==1 and abs(q-center)==1:
                            dis[m, n] = 3
                        if (abs(k-center)==0 and abs(q-center)==2) or (abs(k-center)==2 and abs(q-center)==0):
                            dis[m, n] = 4
                        if (abs(k-center)==2 and abs(q-center)==1) or (abs(k-center)==1 and abs(q-center)==2):
                            dis[m, n] = 5
                        if  abs(k-center)==2 and abs(q-center)==2:
                            dis[m, n] = 6
    return dis.cuda()

def get_edge_A(patch):
    edge_A=np.zeros((patch*patch,patch*patch,4))
    grid = [[0 for j in range(patch)] for i in range(patch)]
    begin=[0,0]
    target=[0,0]
    for i in range(patch*patch):
        for j in range(patch*patch):
            begin[0] = i // patch
            begin[1] = i % patch
            target[0] = j // patch
            target[1] = j % patch
            edge_A[i,j,:]=path_search(grid,begin,target)
    edge_A=torch.from_numpy(edge_A).cuda()
    edge_A=torch.reshape(edge_A,(patch*patch*patch*patch,4))
    M = torch.sum(edge_A, dim=1)
    for i in range(patch*patch*patch*patch):
        if M[i]!=0:
            edge_A[i,:]=edge_A[i,:]/ M[i]
    edge_A=torch.reshape(edge_A,(patch*patch,patch*patch,4))
    return edge_A.type(torch.float32)





