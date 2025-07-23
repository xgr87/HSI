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

%% ����
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
    %% ѡȡ����
    ratio = 0.1;   % ѵ����������
    %     tationum = 5;  % ѵ����������/ÿ��
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
    traindata = reshape(traindata',1,dim,1,ntrain); %4������(1,dim,1,n)�ֱ����ͼ��ģ�������ͨ��������������
    trainlabels=categorical(trainlabels); % ������Ҫ���ǩ������categorical
    
    ntest = size(testdata,1);
    testdata = reshape(testdata',1,dim,1,ntest); %4������(1,dim,1,n)�ֱ����ͼ��ģ�������ͨ��������������
    testlabels=categorical(testlabels); % ������Ҫ���ǩ������categorical
    
    analyzeNetwork(layers)
    net = trainNetwork(traindata,trainlabels,layers,options);
    %% ����
    plabel = classify(net,testdata); %�������
    plabel =double(plabel); %ת��Ϊ����ʾ�ı�ǩ
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