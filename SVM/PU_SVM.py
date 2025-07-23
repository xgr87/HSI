import torch
import numpy as np
import time
from sklearn.svm import SVC
from scipy.io import loadmat
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def classficationresult(testlabel, truelabel):
    n = len(truelabel)
    labelset = np.unique(truelabel)
    classnumber = len(labelset)
    classmatrix = np.zeros((classnumber, classnumber))

    for i in range(classnumber):
        pos = np.where(truelabel == labelset[i])[0]
        midlabel = testlabel[pos]
        num = np.zeros(classnumber)
        for j in range(classnumber):
            num[j] = np.sum(midlabel == labelset[j])
        classmatrix[i, :] = num
    diagelement = np.diag(classmatrix)
    sumdiag = np.sum(diagelement)
    OA = sumdiag / n * 100

    accurperclass = diagelement / np.sum(classmatrix, axis=1) * 100
    AA = np.mean(accurperclass)

    Rcsum = np.sum(np.sum(classmatrix, axis=1) * np.sum(classmatrix, axis=0))
    K = (n * sumdiag - Rcsum) / (n * n - Rcsum)

    return OA, AA, K, accurperclass

dir = './datasets/'
data = loadmat(dir + 'PaviaU.mat')
data_gt = loadmat(dir + 'PaviaU_gt.mat')
spectral = data['paviaU']
spectral_gt = data_gt['paviaU_gt']

# 下采样
spectral = spectral[:, :, ::10]

[n, m, dim] = spectral.shape

# data normalization
X = spectral.reshape(-1, spectral.shape[-1])
Xlabel = spectral_gt.reshape(-1, 1).flatten()
X = (X - np.min(X)) / (np.max(X) - np.min(X))
# X = torch.tensor(X, dtype=torch.float32)

# # 自动计算保留99%方差所需的主成分数量
# pca = PCA(n_components=0.99)
# pca.fit(X)

# # 应用PCA变换
# X = pca.transform(X)

numberoflabel = np.max(Xlabel)
numberofdata = np.zeros(numberoflabel)

num = 10
OA = np.zeros(num)
AA = np.zeros(num)
K = np.zeros(num)
accurperclass = np.zeros((numberoflabel, num))
plabel_SVM = np.zeros((n, m, num))

for randi in range(num):
    loop_start = time.time()  # 单次实验开始计时
    
    # 选取样本
    ratio = 0.1  # 训练样本比例
    traindataNo = []
    testdataNo = []
    for i in range(1, numberoflabel + 1):
        ind = np.where(Xlabel == i)[0]
        numberofdata[i - 1] = len(ind)
        if numberofdata[i - 1] != 0:
            numperclass = int(numberofdata[i - 1])
            No = np.random.permutation(numperclass)
            Numper = int(np.ceil(numberofdata[i - 1] * ratio))
            traindataNo.extend(ind[No[:Numper]])
            testdataNo.extend(ind[No[Numper:]])
    traindata = X[traindataNo]
    testdata = X[testdataNo]
    trainlabels = Xlabel[traindataNo]
    testlabels = Xlabel[testdataNo]

    # 初始化LDA
    lda = LinearDiscriminantAnalysis().fit(traindata, trainlabels)
    # 应用变换
    traindata = lda.transform(traindata)
    testdata = lda.transform(testdata)

    # SVM classification with spectral data
    parameters = {'kernel': ['rbf'], 'C': np.logspace(-3, 3, 7), 'gamma': np.logspace(-3, 3, 7)}
    svm = SVC()
    clf = GridSearchCV(svm, parameters, n_jobs=-1)
    clf.fit(traindata, trainlabels)
    # Test the model
    plabel = clf.predict(testdata)
    accOSVM = accuracy_score(testlabels, plabel)
    label_SVM = Xlabel.copy()
    label_SVM[testdataNo] = plabel
    plabel_SVM[:,:,randi] = label_SVM.reshape(spectral.shape[:-1])
    OA[randi], AA[randi], K[randi], accurperclass[:, randi] = classficationresult(plabel, testlabels)

    # 打印单次实验耗时
    loop_time = time.time() - loop_start
    print(f"实验 {randi + 1}/{num} 耗时: {loop_time:.2f}秒")

# ===== 计算并打印平均指标 =====
mean_OA = np.mean(OA)
mean_AA = np.mean(AA)
mean_Kappa = np.mean(K)

print("\n==== 10次实验综合结果 ====")
print(f"维数: {dim}")
print(f"平均 OA: {mean_OA:.2f}%")
print(f"平均 AA: {mean_AA:.2f}%")
print(f"平均 Kappa系数: {mean_Kappa:.2f}")

# np.savez(
#     dir + 'PaviaU_SVM.npz', OA=OA, AA=AA, K=K,
#     accurperclass=accurperclass, plabel_SVM=plabel_SVM)
# torch.save([OA, AA, K, accurperclass, plabel_SVM, plabel_SVM], dir + 'paviaU_SVM.pt')