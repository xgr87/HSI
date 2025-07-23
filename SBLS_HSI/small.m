% ==== PCANet Demo =======
clear all; close all; clc; 
addpath('./Utils');
addpath('./Liblinear');
%% 
% load data
load('./Indian_pines/Indian_pines_corrected'); 
load('./Indian_pines/Indian_pines_gt'); 
% =======================================
Mi=min(min(min(indian_pines_corrected)));
Ma=max(max(max(indian_pines_corrected)));
for i=1:size(indian_pines_corrected,1)
    for j=1:size(indian_pines_corrected,2)
         indian_pines_corrected(i,j,:)=(indian_pines_corrected(i,j,:)-Mi)/(Ma-Mi);
    end
end
[C D]=find(indian_pines_gt>0);

TestingAccuracy=zeros(1,10);
TestingAccuracy_o=zeros(1,10);
for xxx=1:10
NClass=16;
TrnLabels=[];
TestLabels=[];
Tr_idx_C=[];
Te_idx_C=[];
Te_idx_R=[];
Tr_idx_R=[];
for i=1:NClass   
    [R C]=find(indian_pines_gt==i);
%  Num=ceil(numel(C)*0.1);
    if numel(C)>40
        Num=30;
    else
        Num=ceil(numel(C)*0.5);
    end
    idx_rand=randperm(numel(C));
    Tr_idx_C=[Tr_idx_C C(idx_rand(1:Num))'];
    Tr_idx_R=[Tr_idx_R R(idx_rand(1:Num))'];
    Te_idx_R=[Te_idx_R R(idx_rand(Num+1:end))'];
    Te_idx_C=[Te_idx_C C(idx_rand(Num+1:end))'];
    TrnLabels=[TrnLabels ones(1,Num)*i];
    TestLabels=[TestLabels ones(1,numel(C)-Num)*i]; 
end
% ===========================================================
nTestImg = length(TestLabels);
%% DLNet parameters (they should be funed based on validation set; i.e., ValData & ValLabel)
% We use the parameters in our IEEE TPAMI submission
DLNet.NumStages = 4;
DLNet.PatchSize = [1 25 1 11];
DLNet.NumFilters = [55 6 50 15];
%DLNet.HistBlockSize = [7 7];
fprintf('\n ====== DLNet Parameters ======= \n')
DLNet

%% PCANet Training with 10000 samples

fprintf('\n ======DLNet Training ======= \n')


tic;
[f,V] = DLNet_train(indian_pines_corrected,TrnLabels,Tr_idx_R,Tr_idx_C,DLNet); % 
ftrain=zeros(size(f,3),numel(Tr_idx_R));
for i=1:numel(Tr_idx_R)
    ftrain(:,i)=reshape(f(Tr_idx_R(i),Tr_idx_C(i),:),size(f,3),1);
end

PCANet_TrnTime = toc;
clear TrnData_ImgCell; 

fprintf('\n ====== Training Classifier ======= \n')
tic;
LinearSVM_TrnTime = toc;

%% PCANet Feature Extraction and Testing 
fprintf('\n ====== MFANet Testing ======= \n')
TestData=zeros(200,size(Te_idx_C,2));


nCorrRecog = 0;
RecHistory = zeros(numel(Te_idx_C),1);

tic; 
ft = MFANet_Test(indian_pines_corrected,V,DLNet);
ftest=zeros(size(ft,3),numel(Te_idx_R));
for i=1:numel(Te_idx_R)
    ftest(:,i)=reshape(ft(Te_idx_R(i),Te_idx_C(i),:),size(ft,3),1);
end

pa1=1000;
pa2=1;

tic
[TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy(xxx),label_index_expected(xxx,:)] = elm_kernel(ftrain, TrnLabels,ftest,TestLabels,1, pa1, 'RBF_kernel',pa2);
toc

[kappa(xxx), acc(:,xxx), acc_O(xxx), acc_A(xxx)] = evaluate_results(label_index_expected(xxx,:), TestLabels);
end






    