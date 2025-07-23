% ==== PCANet Demo =======
clear all; close all; clc; 
addpath('./Utils');
addpath('./Liblinear');
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
Nm=1;
for K1=10:10:80
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
 Num=ceil(numel(C)*0.1);
    idx_rand=randperm(numel(C));
    Tr_idx_C=[Tr_idx_C C(idx_rand(1:Num))'];
    Tr_idx_R=[Tr_idx_R R(idx_rand(1:Num))'];
    Te_idx_R=[Te_idx_R R(idx_rand(Num+1:end))'];
    Te_idx_C=[Te_idx_C C(idx_rand(Num+1:end))'];
    TrnLabels=[TrnLabels ones(1,Num)*i];
    TestLabels=[TestLabels ones(1,numel(C)-Num)*i]; 
end

TrnData=zeros(200,numel(Tr_idx_C));
for i=1:numel(Tr_idx_C)
    TrnData(:,i) = reshape(indian_pines_corrected(Tr_idx_R(i),Tr_idx_C(i),:),200,1); % 
end

nTestImg = length(TestLabels);
PCANet.NumStages = 2;
PCANet.PatchSize = 21;
PCANet.NumFilters = [K1 5];
PCANet.HistBlockSize = [7 7];
PCANet.BlkOverLapRatio = 0.5;
fprintf('\n ====== PCANet Parameters ======= \n')
PCANet

%% PCANet Training with 10000 samples

fprintf('\n ====== PCANet Training ======= \n')
tic;
[ftrain,V] = LDANet_train(TrnData,indian_pines_corrected,TrnLabels,Tr_idx_R,Tr_idx_C,PCANet,1); % BlkIdx serves the purpose of learning block-wise DR projection matrix; e.g., WPCA
PCANet_TrnTime = toc;
clear TrnData_ImgCell; 
fprintf('\n ====== Training Linear SVM Classifier ======= \n')
tic;
LinearSVM_TrnTime = toc;

%% PCANet Feature Extraction and Testing 
fprintf('\n ====== PCANet Testing ======= \n')
TestData=zeros(200,size(Te_idx_C,2));
nCorrRecog = 0;
RecHistory = zeros(numel(Te_idx_C),1);

tic; 
ftest = LDANet_FeaExt_new(indian_pines_corrected,Te_idx_R,Te_idx_C,V,PCANet);
tic
[TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy(xxx),label_index_expected] = elm_kernel(ftrain, TrnLabels,ftest,TestLabels,1, 1000, 'RBF_kernel',45);
toc
[kappa(Nm,xxx), acc(:,xxx), acc_O(Nm,xxx), acc_A(Nm,xxx)] = evaluate_results(label_index_expected, TestLabels);
end
Nm=Nm+1;
end

save Result_7.19_K1_5

% ==== PCANet Demo =======
clear all; close all; clc; 
addpath('./Utils');
addpath('./Liblinear');
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
Nm=1;
for K1=10:10:80
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
 Num=ceil(numel(C)*0.1);
    idx_rand=randperm(numel(C));
    Tr_idx_C=[Tr_idx_C C(idx_rand(1:Num))'];
    Tr_idx_R=[Tr_idx_R R(idx_rand(1:Num))'];
    Te_idx_R=[Te_idx_R R(idx_rand(Num+1:end))'];
    Te_idx_C=[Te_idx_C C(idx_rand(Num+1:end))'];
    TrnLabels=[TrnLabels ones(1,Num)*i];
    TestLabels=[TestLabels ones(1,numel(C)-Num)*i]; 
end

TrnData=zeros(200,numel(Tr_idx_C));
for i=1:numel(Tr_idx_C)
    TrnData(:,i) = reshape(indian_pines_corrected(Tr_idx_R(i),Tr_idx_C(i),:),200,1); % 
end

nTestImg = length(TestLabels);
PCANet.NumStages = 2;
PCANet.PatchSize = 21;
PCANet.NumFilters = [K1 10];
PCANet.HistBlockSize = [7 7];
PCANet.BlkOverLapRatio = 0.5;
fprintf('\n ====== PCANet Parameters ======= \n')
PCANet

%% PCANet Training with 10000 samples

fprintf('\n ====== PCANet Training ======= \n')
tic;
[ftrain,V] = LDANet_train(TrnData,indian_pines_corrected,TrnLabels,Tr_idx_R,Tr_idx_C,PCANet,1); % BlkIdx serves the purpose of learning block-wise DR projection matrix; e.g., WPCA
PCANet_TrnTime = toc;
clear TrnData_ImgCell; 
fprintf('\n ====== Training Linear SVM Classifier ======= \n')
tic;
LinearSVM_TrnTime = toc;

%% PCANet Feature Extraction and Testing 
fprintf('\n ====== PCANet Testing ======= \n')
TestData=zeros(200,size(Te_idx_C,2));
nCorrRecog = 0;
RecHistory = zeros(numel(Te_idx_C),1);

tic; 
ftest = LDANet_FeaExt_new(indian_pines_corrected,Te_idx_R,Te_idx_C,V,PCANet);
tic
[TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy(xxx),label_index_expected] = elm_kernel(ftrain, TrnLabels,ftest,TestLabels,1, 1000, 'RBF_kernel',45);
toc
[kappa(Nm,xxx), acc(:,xxx), acc_O(Nm,xxx), acc_A(Nm,xxx)] = evaluate_results(label_index_expected, TestLabels);
end
Nm=Nm+1;
end

save Result_7.19_K1_10

% ==== PCANet Demo =======
clear all; close all; clc; 
addpath('./Utils');
addpath('./Liblinear');
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
Nm=1;
for K1=10:10:80
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
 Num=ceil(numel(C)*0.1);
    idx_rand=randperm(numel(C));
    Tr_idx_C=[Tr_idx_C C(idx_rand(1:Num))'];
    Tr_idx_R=[Tr_idx_R R(idx_rand(1:Num))'];
    Te_idx_R=[Te_idx_R R(idx_rand(Num+1:end))'];
    Te_idx_C=[Te_idx_C C(idx_rand(Num+1:end))'];
    TrnLabels=[TrnLabels ones(1,Num)*i];
    TestLabels=[TestLabels ones(1,numel(C)-Num)*i]; 
end

TrnData=zeros(200,numel(Tr_idx_C));
for i=1:numel(Tr_idx_C)
    TrnData(:,i) = reshape(indian_pines_corrected(Tr_idx_R(i),Tr_idx_C(i),:),200,1); % 
end

nTestImg = length(TestLabels);
PCANet.NumStages = 2;
PCANet.PatchSize = 21;
PCANet.NumFilters = [K1 15];
PCANet.HistBlockSize = [7 7];
PCANet.BlkOverLapRatio = 0.5;
fprintf('\n ====== PCANet Parameters ======= \n')
PCANet

%% PCANet Training with 10000 samples

fprintf('\n ====== PCANet Training ======= \n')
tic;
[ftrain,V] = LDANet_train(TrnData,indian_pines_corrected,TrnLabels,Tr_idx_R,Tr_idx_C,PCANet,1); % BlkIdx serves the purpose of learning block-wise DR projection matrix; e.g., WPCA
PCANet_TrnTime = toc;
clear TrnData_ImgCell; 
fprintf('\n ====== Training Linear SVM Classifier ======= \n')
tic;
LinearSVM_TrnTime = toc;

%% PCANet Feature Extraction and Testing 
fprintf('\n ====== PCANet Testing ======= \n')
TestData=zeros(200,size(Te_idx_C,2));
nCorrRecog = 0;
RecHistory = zeros(numel(Te_idx_C),1);

tic; 
ftest = LDANet_FeaExt_new(indian_pines_corrected,Te_idx_R,Te_idx_C,V,PCANet);
tic
[TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy(xxx),label_index_expected] = elm_kernel(ftrain, TrnLabels,ftest,TestLabels,1, 1000, 'RBF_kernel',45);
toc
[kappa(Nm,xxx), acc(:,xxx), acc_O(Nm,xxx), acc_A(Nm,xxx)] = evaluate_results(label_index_expected, TestLabels);
end
Nm=Nm+1;
end

save Result_7.19_K1_15

% ==== PCANet Demo =======
clear all; close all; clc; 
addpath('./Utils');
addpath('./Liblinear');
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
Nm=1;
for K1=10:10:80
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
 Num=ceil(numel(C)*0.1);
    idx_rand=randperm(numel(C));
    Tr_idx_C=[Tr_idx_C C(idx_rand(1:Num))'];
    Tr_idx_R=[Tr_idx_R R(idx_rand(1:Num))'];
    Te_idx_R=[Te_idx_R R(idx_rand(Num+1:end))'];
    Te_idx_C=[Te_idx_C C(idx_rand(Num+1:end))'];
    TrnLabels=[TrnLabels ones(1,Num)*i];
    TestLabels=[TestLabels ones(1,numel(C)-Num)*i]; 
end

TrnData=zeros(200,numel(Tr_idx_C));
for i=1:numel(Tr_idx_C)
    TrnData(:,i) = reshape(indian_pines_corrected(Tr_idx_R(i),Tr_idx_C(i),:),200,1); % 
end

nTestImg = length(TestLabels);
PCANet.NumStages = 2;
PCANet.PatchSize = 21;
PCANet.NumFilters = [K1 20];
PCANet.HistBlockSize = [7 7];
PCANet.BlkOverLapRatio = 0.5;
fprintf('\n ====== PCANet Parameters ======= \n')
PCANet

%% PCANet Training with 10000 samples

fprintf('\n ====== PCANet Training ======= \n')
tic;
[ftrain,V] = LDANet_train(TrnData,indian_pines_corrected,TrnLabels,Tr_idx_R,Tr_idx_C,PCANet,1); % BlkIdx serves the purpose of learning block-wise DR projection matrix; e.g., WPCA
PCANet_TrnTime = toc;
clear TrnData_ImgCell; 
fprintf('\n ====== Training Linear SVM Classifier ======= \n')
tic;
LinearSVM_TrnTime = toc;

%% PCANet Feature Extraction and Testing 
fprintf('\n ====== PCANet Testing ======= \n')
TestData=zeros(200,size(Te_idx_C,2));
nCorrRecog = 0;
RecHistory = zeros(numel(Te_idx_C),1);

tic; 
ftest = LDANet_FeaExt_new(indian_pines_corrected,Te_idx_R,Te_idx_C,V,PCANet);
tic
[TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy(xxx),label_index_expected] = elm_kernel(ftrain, TrnLabels,ftest,TestLabels,1, 1000, 'RBF_kernel',45);
toc
[kappa(Nm,xxx), acc(:,xxx), acc_O(Nm,xxx), acc_A(Nm,xxx)] = evaluate_results(label_index_expected, TestLabels);
end
Nm=Nm+1;
end

save Result_7.19_K1_20



    