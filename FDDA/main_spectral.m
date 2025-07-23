clc,clear

dir = 'E:\函数型数据分析\Program\data\';
addpath(genpath(pwd))

load([dir,'Indian_pines_corrected.mat']);
load([dir,'Indian_pines_gt.mat']);
spectral = indian_pines_corrected;
[n,m,dim] = size(spectral);
Xlabel = double(reshape(indian_pines_gt,n*m,1));

% load([dir,'Salinas_corrected.mat']);
% load([dir,'Salinas_gt.mat']);
% spectral = salinas_corrected;
% [n,m,dim] = size(spectral);
% Xlabel = double(reshape(salinas_gt,n*m,1));

% load([dir,'PaviaU.mat']);
% load([dir,'PaviaU_gt.mat']);
% spectral = paviaU;
% [n,m,dim] = size(spectral);
% Xlabel = double(reshape(paviaU_gt,n*m,1));

%% data normalization
X = reshape(spectral,n*m,dim);
X = mapminmax(X);

%% claasification results
OA=struct(); AA=struct();
K=struct();accurperclass=struct();

parpool(4);
% tic;

numberoflabel = max(Xlabel);
for randi = 1 : 10
    %% 选取样本
    ratio = 0.1;   % 训练样本比例
    % tationum = 10;  % 训练样本个数/每类
    traindataNo=[];testdataNo=[];
    trainlabels = []; testlabels = [];
    numberofdata = zeros(1,numberoflabel);
    Labelmatrix = zeros(n,m);
    for i = 1:numberoflabel
        ind = find(Xlabel==i);
        numberofdata(i) = length(ind);
        if(numberofdata(i) ~= 0)
            No = randperm(numberofdata(i));
            Numper = ceil(numberofdata(i)*ratio);
            traindataNo = [traindataNo;ind(No(1:Numper))];
            testdataNo = [testdataNo;ind(No(Numper+1:numberofdata(i)))];
        end
    end
    traindata = X(traindataNo,:);
    testdata = X(testdataNo,:);
    
    trainlabels = Xlabel(traindataNo);
    testlabels= Xlabel(testdataNo);
    
%     testlabels = [trainlabels;testlabels];
%     testdataNo = [traindataNo;testdataNo];
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [numtrain,~] = size(traindata);   %训练样本大小和维数
    [numtest,~] = size(testdata);      %测试样本大小和维数
    sample = [traindata; testdata]';
    num = numtrain + numtest;
    %% functional data fit
    % 生成B样条基函数
    r = 10^(-3);
    rangewavelength = r * [1, dim]';  %波长取值范围
    norder = 4;                % B样条阶数
    % 连续分布的数据
    wavelength = r * (1:dim)';   % 波长
    nbasis = norder + length(wavelength) - 2;
    Lfd = 2;   %Lfd为粗糙惩罚项的导数的阶数
    
    basisobj = create_bspline_basis(rangewavelength, nbasis, norder, wavelength);
    
    %  名称：定义域，物质，辐射值
    fdnames{1} = 'Wavelength';
    fdnames{2} = 'Substance';
    fdnames{3} = 'Radiance';
    
    % 选取lambda
    lnlam = -8:1:0;    %lambda的取值范围
    gcvsave = zeros(length(lnlam),1);
    parfor i=1:length(lnlam)
        fdParobj = fdPar(basisobj, Lfd, 10^lnlam(i));
        % 计算拟合误差gcv
        [~, ~, gcv] = smooth_basis(wavelength, sample, fdParobj, [], fdnames);   %the default is 'method' =  'chol', but if 'method' = 'qr', the qr decomposition is used.
        gcvsave(i) = sum(gcv);
    end
    [~, k1] = max(-gcvsave);   %计算所有样本拟合误差最小的lambda。
    lambda = 10^lnlam(k1);  % 最优lambda
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 粗糙惩罚法
    fdParobj = fdPar(basisobj, Lfd, lambda);
    [fdobj, df, gcv] = smooth_basis(wavelength, sample, fdParobj, [], fdnames);
    fdmat = eval_fd(wavelength, fdobj);    %获取拟合之后的数据
    coef   = getcoef(fdobj);  %取基函数的系数
    %% SVM classification with spectral-spatial original data
    
    Odatatrain = sample(:,1:numtrain)';          % only uses the spectral information
    Odatatest  = sample(:,1+numtrain : num)';

    %     [~, bestc, bestg] = SVMcgForClass(trainlabels,Odatatrain,-10,10,-10,10,10,1,1);
    %     cmd0 = ['-c ', num2str(bestc), ' -g ', num2str(bestg), ' -t ', num2str(2)];
    %     cmd0 = ['-c ', num2str(10), ' -g ', num2str(0.1), ' -t ', num2str(2)];
    cmd0 = ['-c ', num2str(1000), ' -g ', num2str(1), ' -t ', num2str(2)];   % only uses the spectral information
    model = svmtrain(trainlabels, Odatatrain, cmd0);
    type = 2;
    CR0 = ClassResult(testlabels, Odatatest, model, type);  %测试样本分类结果
    
    accOSVM(randi) = CR0.accuracy(1);  %精度
    label_OSVM = Xlabel;
    label_OSVM(testdataNo) = CR0.plabel;
    plabel_OSVM(:,:,randi) = reshape(label_OSVM,n,m);
    [OA.OSVM(randi),AA.OSVM(randi),K.OSVM(randi),accurperclass.OSVM(:,randi)] = classficationresult(CR0.plabel,testlabels);
    clear model;
    %      %% SVM classification after denoising
    %     Ddatatrain = fdmatT(:,1:numtrain)';
    %     Ddatatest  = fdmatT(:,1+numtrain : num)';
    %
    %     %     [~, bestc, bestg] = SVMcgForClass(trainlabels,fdatatrain,-10,10,-10,10,10,1,1);
    %     %     cmd1 = ['-c ', num2str(bestc), ' -g ', num2str(bestg), ' -t ', num2str(2)];
    %     cmd01 = ['-c ', num2str(100), ' -g ', num2str(0.1), ' -t ', num2str(2)];
    %     model = svmtrain(trainlabels, Ddatatrain, cmd01);
    %     type = 2;
    %     CR01 = ClassResult(testlabels, Ddatatest, model, type);
    %     accDSVM(randi) = CR01.accuracy(1);  %精度
    %     label_DSVM = Xlabel;
    %     label_DSVM(testdataNo) = CR01.plabel;
    %     plabel_DSVM(:,:,randi) = reshape(label_DSVM,n,m);
    %     [OA.DSVM(randi),AA.DSVM(randi),K.DSVM(randi),accurperclass.DSVM(:,randi)] = classficationresult(CR01.plabel,testlabels);
    %% FSVM classification with spectral-spatial functional data
    fdatatrain = coef(:,1:numtrain)';      % only uses the spectral information
    fdatatest  = coef(:,1+numtrain : num)';
    %     [~, bestc, bestg] = SVMcgForClass(trainlabels,fdatatrain,-10,10,-10,10,10,1,1);
    %     cmd1 = ['-c ', num2str(bestc), ' -g ', num2str(bestg), ' -t ', num2str(2)];
    %     cmd1 = ['-c ', num2str(100), ' -g ', num2str(1), ' -t ', num2str(2)];
    cmd1 = ['-c ', num2str(100), ' -g ', num2str(1), ' -t ', num2str(2)];  % only uses the spectral information
    model = svmtrain(trainlabels, fdatatrain, cmd1);
    type = 2;
    CR1 = ClassResult(testlabels, fdatatest, model, type);
    accFSVM(randi) = CR1.accuracy(1);  %精度
    label_FSVM = Xlabel;
    label_FSVM(testdataNo) = CR1.plabel;
    plabel_FSVM(:,:,randi) = reshape(label_FSVM,n,m);
    [OA.FSVM(randi),AA.FSVM(randi),K.FSVM(randi),accurperclass.FSVM(:,randi)] = classficationresult(CR1.plabel,testlabels);
    %% PCA+SVM
    [COEFF,SCORE] = pca(sample');   % only uses the spectral information
    PC = SCORE(:,1:37);
    train_final = PC(1:numtrain, :);  %训练样本的函数型特征
    test_final = PC(numtrain + 1:num, :);   %测试样本的函数型特征
    
    %svm分类
    %     [~, bestc, bestg] = SVMcgForClass(trainlabels,train_final,-10,10,-10,10,5,1,1);
    %     cmd2 = ['-c ', num2str(bestc), ' -g ', num2str(bestg), ' -t ', num2str(2)];
    %     cmd2 = ['-c ', num2str(100), ' -g ', num2str(1), ' -t ', num2str(2)];
    cmd2 = ['-c ', num2str(100), ' -g ', num2str(1), ' -t ', num2str(2)];
    model = svmtrain(trainlabels, train_final, cmd2);
    type = 2;
    CR2 = ClassResult(testlabels, test_final, model, type);
    accPCA(randi) = CR2.accuracy(1);  %精度
    label_PCA = Xlabel;
    label_PCA(testdataNo) = CR2.plabel;
    plabel_PCA(:,:,randi) = reshape(label_PCA,n,m);
    [OA.PCA(randi),AA.PCA(randi),K.PCA(randi),accurperclass.PCA(:,randi)] = classficationresult(CR2.plabel,testlabels);
    %% FPCA+SVM
    pcaf = pca_fd(fdobj, 34);   %函数型PCA
    train_final = pcaf.harmscr(1:numtrain, :);  %训练样本的函数型特征
%     test_final = pcaf.harmscr(numtrain + 1:num, :);   %测试样本的函数型特征
    test_final = pcaf.harmscr;   %测试样本的函数型特征
    %svm分类
    %     [~, bestc, bestg] = SVMcgForClass(trainlabels,train_final,-10,10,-10,10,10,1,1);
    %     cmd3 = ['-c ', num2str(bestc), ' -g ', num2str(bestg), ' -t ', num2str(2)];
    %     cmd3 = ['-c ', num2str(10^5), ' -g ', num2str(1), ' -t ', num2str(2)];
    cmd3 = ['-c ', num2str(100), ' -g ', num2str(1000), ' -t ', num2str(2)];
    model = svmtrain(trainlabels, train_final, cmd3);
    type = 2;
    CR3 = ClassResult(testlabels, test_final, model, type);
    accFPCA(randi) = CR3.accuracy(1);  %精度
    label_FPCA = Xlabel;
    label_FPCA(testdataNo) = CR3.plabel;
    plabel_FPCA(:,:,randi) = reshape(label_FPCA,n,m);
    [OA.FPCA(randi),AA.FPCA(randi),K.FPCA(randi),accurperclass.FPCA(:,randi)] = classficationresult(CR3.plabel,testlabels);
    %% LDA + SVM
    Ospectraltrain = sample(:,1:numtrain)';
    Ospectraltest  = sample(:,1+numtrain : num)';
    
    options.Regu = 1;
    [eigvector,~] = LDA(Ospectraltrain, trainlabels, options);
    OTrain_data = Ospectraltrain*eigvector;
    OTest_data = Ospectraltest*eigvector;
    clear eigvector;
    
    
    OTrain_feature = OTrain_data;
    OTest_feature = OTest_data;
    % svm分类
    %     [~, bestc, bestg] = SVMcgForClass(trainlabels,OTrain_data,-10,10,-10,10,10,1,1);
    %     cmd4 = ['-c ', num2str(bestc), ' -g ', num2str(bestg), ' -t ', num2str(2)];
    %     cmd4 = ['-c ', num2str(20), ' -g ', num2str(1), ' -t ', num2str(2)];
    cmd4 = ['-c ', num2str(20), ' -g ', num2str(30), ' -t ', num2str(2)];
    model = svmtrain(trainlabels, OTrain_feature, cmd4);
    type = 2;
    CR4 = ClassResult(testlabels, OTest_feature, model, type);
    accLDA(randi) = CR4.accuracy(1);  %精度
    label_LDA = Xlabel;
    label_LDA(testdataNo) = CR4.plabel;
    plabel_LDA(:,:,randi) = reshape(label_LDA,n,m);
    [OA.LDA(randi),AA.LDA(randi),K.LDA(randi),accurperclass.LDA(:,randi)] = classficationresult(CR4.plabel,testlabels);
    %% FDDA + SVM
    spectraltrain = coef(:,1:numtrain)';
    spectraltest  = coef(:,1+numtrain : num)';
    
    options.Regu = 1;
    [eigvector,~] = LDA(spectraltrain, trainlabels, options);
    Train_data = spectraltrain*eigvector;
    Test_data = spectraltest*eigvector;
    clear eigvector;
    
    Train_feature = Train_data;
    Test_feature = Test_data;
    % svm分类
    %     [bestCVaccuracy, bestc, bestg] = SVMcgForClass(trainlabels,Train_data,-10,10,-10,10,10,1,1);
    %     cmd5 = ['-c ', num2str(bestc), ' -g ', num2str(bestg), ' -t ', num2str(2)];
    %     cmd5 = ['-c ', num2str(10), ' -g ', num2str(100), ' -t ', num2str(2)];
    cmd5 = ['-c ', num2str(16), ' -g ', num2str(512), ' -t ', num2str(2)];
    model = svmtrain(trainlabels, Train_feature, cmd5);
    type = 2;
    CR5 = ClassResult(testlabels, Test_feature, model, type);
    accFDA(randi) = CR5.accuracy(1);  %精度
    label_FDA = Xlabel;
    label_FDA(testdataNo) = CR5.plabel;
    plabel_FDA(:,:,randi) = reshape(label_FDA,n,m);
    [OA.FDA(randi),AA.FDA(randi),K.FDA(randi),accurperclass.FDA(:,randi)] = classficationresult(CR5.plabel,testlabels);
    
    clear traindataNo testdataNo
end

% save Sa_1.mat XX;
% save SalinasP.mat GroundT img
% save PU_1.mat XX;
% save PaviaUP.mat GroundT img
% save indian_result.mat plabel_OSVM plabel_FSVM plabel_PCA plabel_FPCA plabel_LDA plabel_FDA OA AA K accurperclass
% toc;
delete(gcp);