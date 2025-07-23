clc,clear

dir = '.\datasets\';
addpath(genpath(pwd))
sr = 1;
load([dir,'Indian_pines.mat']);
load([dir,'Indian_pines_gt.mat']);
spectral = indian_pines(:,:,1:sr:end);
% spectral = indian_pines(:,:,[1:103,109:149,164:219]); %including water absorption bands [104-108], [150-163], 220 = [1:103,109:149,164:219]
[n,m,dim] = size(spectral);
Xlabel = double(reshape(indian_pines_gt,n*m,1));
clear indian_pines indian_pines_gt;

% parpool(4);
tic;
%% data normalization
X = reshape(spectral,n*m,dim);
X = mapminmax(X);
numberoflabel = max(Xlabel);
numberofdata = zeros(1,numberoflabel);
%% claasification results
num = 10;
OA=zeros(1,num); AA=OA;
Ka=OA;accurperclass=zeros(numberoflabel,num);

%% 网络搭建
layers = [ ...
    imageInputLayer([1 dim 1])       %%2D-CNN
    
    convolution2dLayer([1,650],20)     %kernel [1,24],20
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer([1 2],'Stride',2)
    
    convolution2dLayer([1,220],30)     %kernel [1,24],20
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer([1 2],'Stride',2)
    
    fullyConnectedLayer(numberoflabel)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'MaxEpochs',100,...
    'MiniBatchSize',27, ...
    'GradientThreshold',1, ...
    'Verbose',false, ...
    'Plots','training-progress');

%% Training and Test
% rng(0)
for randi = 1 : num
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% 选取样本
    ratio = 0.1;   % 训练样本比例
    %     tationum = 5;  % 训练样本个数/每类
    traindataNo=[];testdataNo=[];
    for i = 1:numberoflabel
        ind = find(Xlabel==i);
        numberofdata(i) = length(ind);
        if(numberofdata(i) ~= 0)
            No = randperm(numberofdata(i));
            Numper = ceil(numberofdata(i)*ratio);
            %             Numper = tationum;
            traindataNo = [traindataNo;ind(No(1:Numper))];
            testdataNo = [testdataNo;ind(No(Numper+1:numberofdata(i)))];
        end
    end
    traindata = X(traindataNo,:);
    testdata = X(testdataNo,:);
    
    trainlabels = Xlabel(traindataNo);
    testlabels= Xlabel(testdataNo);
    
    %% CNN classification with spectral data
    ntrain = size(traindata,1);
    traindata = reshape(traindata',1,dim,1,ntrain); %4个参数(1,dim,1,n)分别代表图像的（长，宽，通道数，样本量）
    trainlabels=categorical(trainlabels); % 函数包要求标签类型是categorical
    
    ntest = size(testdata,1);
    testdata = reshape(testdata',1,dim,1,ntest); %4个参数(1,dim,1,n)分别代表图像的（长，宽，通道数，样本量）
    testlabels=categorical(testlabels); % 函数包要求标签类型是categorical
    
    analyzeNetwork(layers)
    net = trainNetwork(traindata,trainlabels,layers,options);
    %% 测试
    plabel = classify(net,testdata); %网络测试
    plabel =double(plabel); %转化为可显示的标签
    testlabels = double(testlabels);
    accCNN(randi) = sum(plabel == testlabels)/numel(testlabels);
    
    label_CNN = Xlabel;
    label_CNN(testdataNo) = plabel;
    plabel_CNN(:,:,randi) = reshape(label_CNN,n,m);
    [OA(randi),AA(randi),Ka(randi),accurperclass(:,randi)] = classficationresult(plabel,testlabels);
    clear net;
    [OA(randi),AA(randi),Ka(randi)]
end
toc;
% save indian_1DCNN.mat OA AA Ka accurperclass plabel_CNN;
% delete(gcp);