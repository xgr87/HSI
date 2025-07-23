clc;
clear;

tic;
dir = '../datasets/';
addpath(genpath(pwd));

%% Load data
% load('HoustonU2018.mat');
% load('HoustonU2018_gt.mat');
% spectral = double(houstonU2018);
% [n,m,dim] = size(spectral);
% Xlabel = double(reshape(houstonU2018_gt,n*m,1));
% clear houstonU2018 houstonU2018_gt

load([dir,'Salinas1020.mat']);
load([dir,'Salinas_gt.mat']);

Results = zeros(3,10);
for sstep = 1 : 1
    
    spectral = salinas(:,:,1:sstep:end);
    [n,m,dim] = size(spectral);
    label = salinas_gt;
    Xlabel = double(reshape(label,n*m,1));

    %% feature extraction
    r = 20;  %number of feature;
    lamda = 0.1;
    [FE_OTVCA]=OTVCA_V3(spectral,r,lamda);
    [n,m,dim1] = size(FE_OTVCA);
    X = reshape(FE_OTVCA,n*m,dim1);
    numberoflabel = max(Xlabel);
    %% claasification results
    OA=struct();
    AA=struct();
    K=struct();
    accurperclass=struct();
    
    looptime = 1;
    for randi = 1 : looptime
        %% trainset selection
        ratio = 0.1;   % the ratio of labeling samples
        traindataNo=[];
        testdataNo=[];
        numberofdata = zeros(1,numberoflabel);
        Numper = zeros(1,numberoflabel);
        for i = 1:numberoflabel
            ind = find(Xlabel==i);
            numberofdata(i) = length(ind);
            if(numberofdata(i) ~= 0)
                No = randperm(numberofdata(i));
                Numper(i) = ceil(numberofdata(i)*ratio);
                traindataNo = [traindataNo;ind(No(1:Numper(i)))];
                testdataNo = [testdataNo;ind(No(Numper(i)+1:numberofdata(i)))];
            end
        end
        traindata = X(traindataNo,:);
        testdata = X(testdataNo,:);
        
        trainlabels = Xlabel(traindataNo);
        testlabels= Xlabel(testdataNo);
        %% classification using SVM
        [~, bestc, bestg] = SVMcgForClass(trainlabels,traindata,-10,10,-10,10,10,1,1);
        paraOriginal(randi,:) = [bestc, bestg];
        cmd = ['-c ', num2str(bestc), ' -g ', num2str(bestg), ' -t ', num2str(2)];
        %     cmd = ['-c ', num2str(1024), ' -g ', num2str(32), ' -t ', num2str(2)];    %houston2018
        model = svmtrain(trainlabels, traindata, cmd);
        type = 2;
        CR = ClassResult(Xlabel, X, model, type);
        label_OTVCA = CR.plabel;
        label_OTVCA(traindataNo) = trainlabels;
        plabel_OTVCA(:,:,randi) = reshape(label_OTVCA,n,m);   %whole map
        [OA.OTVCA(randi),AA.OTVCA(randi),K.OTVCA(randi),accurperclass.OTVCA(:,randi)] = classficationresult(label_OTVCA(testdataNo),testlabels);
        clear model;
    end
    sstep
    Results(:,11-sstep) = [mean(OA.OTVCA);mean(AA.OTVCA);mean(K.OTVCA)];
end
% save houston2018OTVCA.mat plabel_OTVCA OA AA K accurperclass
toc;