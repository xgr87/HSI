% ==== PCANet Demo =======
clear all; close all; clc; 
addpath('./Utils');
addpath('./Liblinear');
% ImgFormat = 'gray'; %'color' or 'gray'
%% Loading data from MNIST Basic (10000 training, 2000 validation, 50000 testing) 
% load data
load('./Indian_pines/Indian_pines_corrected'); 
load('./Indian_pines/Indian_pines_gt'); 
% =======================================
Mi=min(min(min(indian_pines_corrected)));
Ma=max(max(max(indian_pines_corrected)));
for i=1:size(indian_pines_corrected,1)
    for j=1:size(indian_pines_corrected,2)
         indian_pines_corrected(i,j,:)=(indian_pines_corrected(i,j,:)-Mi)/(Ma-Mi);
% tempp=reshape(indian_pines_corrected(i,j,:),1,200);
% indian_pines_corrected(i,j,:)=indian_pines_corrected(i,j,:)/norm(tempp);
    end
end
[C D]=find(indian_pines_gt>0);

% HIdata=indian_pines_corrected(C,D,:);
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
 Num=ceil(numel(C)*0.1);
%     if numel(C)>50
%         Num=50;
%     else
%         Num=10;
%     end
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

% TestData = HIdata(:,TestLabels)';

% ===========================================================

nTestImg = length(TestLabels);

%% PCANet parameters (they should be funed based on validation set; i.e., ValData & ValLabel)
% We use the parameters in our IEEE TPAMI submission
PCANet.NumStages = 2;
PCANet.PatchSize = 17;
PCANet.NumFilters = [50 20];
PCANet.HistBlockSize = [7 7];
PCANet.BlkOverLapRatio = 0.5;
fprintf('\n ====== PCANet Parameters ======= \n')
PCANet

%% PCANet Training with 10000 samples

fprintf('\n ====== PCANet Training ======= \n')
% TrnData_ImgCell = mat2imgcell(TrnData,ImgSize,ImgSize,ImgFormat); % convert columns in TrnData to cells 
% clear TrnData; 
tic;
[ftrain,V] = LDANet_train(TrnData,indian_pines_corrected,TrnLabels,Tr_idx_R,Tr_idx_C,PCANet,1); % BlkIdx serves the purpose of learning block-wise DR projection matrix; e.g., WPCA
PCANet_TrnTime = toc;
clear TrnData_ImgCell; 


fprintf('\n ====== Training Linear SVM Classifier ======= \n')
% for i=1:size(ftrain,2)
%     ftrain(:,i)=ftrain(:,i)/norm(ftrain(:,i));
% end
tic;
% models = train(TrnLabels', sparse(ftrain'), '-s 1 -q'); % we use linear SVM classifier (C = 1), calling libsvm library
LinearSVM_TrnTime = toc;



%% PCANet Feature Extraction and Testing 
fprintf('\n ====== PCANet Testing ======= \n')
TestData=zeros(200,size(Te_idx_C,2));
% for i=1:size(Te_idx_C)
%     TestData(:,i) = reshape(indian_pines_corrected(Te_idx_R(i),Te_idx_C(i),:),200,1); % 
% end
nCorrRecog = 0;
RecHistory = zeros(numel(Te_idx_C),1);

tic; 
ftest = LDANet_FeaExt_new(indian_pines_corrected,Te_idx_R,Te_idx_C,V,PCANet);
% for i=1:size(ftest,2)
%     ftest(:,i)=ftest(:,i)/norm(ftest(:,i));
% end
% for i=1:size(ftrain,2)
%     ftrain(:,i)=ftrain(:,i)/norm(ftrain(:,i));
% end

tic
[TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy(xxx),label_index_expected] = elm_kernel(ftrain, TrnLabels,ftest,TestLabels,1, 1000, 'RBF_kernel',45);
toc

[kappa(xxx), acc(:,xxx), acc_O(xxx), acc_A(xxx)] = evaluate_results(label_index_expected, TestLabels);
end
save Result_7.19_17train

% ==== PCANet Demo =======
clear all; close all; clc; 
addpath('./Utils');
addpath('./Liblinear');
% ImgFormat = 'gray'; %'color' or 'gray'
%% Loading data from MNIST Basic (10000 training, 2000 validation, 50000 testing) 
% load data
load('./Indian_pines/Indian_pines_corrected'); 
load('./Indian_pines/Indian_pines_gt'); 
% =======================================
Mi=min(min(min(indian_pines_corrected)));
Ma=max(max(max(indian_pines_corrected)));
for i=1:size(indian_pines_corrected,1)
    for j=1:size(indian_pines_corrected,2)
         indian_pines_corrected(i,j,:)=(indian_pines_corrected(i,j,:)-Mi)/(Ma-Mi);
% tempp=reshape(indian_pines_corrected(i,j,:),1,200);
% indian_pines_corrected(i,j,:)=indian_pines_corrected(i,j,:)/norm(tempp);
    end
end
[C D]=find(indian_pines_gt>0);

% HIdata=indian_pines_corrected(C,D,:);
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
 Num=ceil(numel(C)*0.1);
%     if numel(C)>50
%         Num=50;
%     else
%         Num=10;
%     end
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

% TestData = HIdata(:,TestLabels)';

% ===========================================================

nTestImg = length(TestLabels);

%% PCANet parameters (they should be funed based on validation set; i.e., ValData & ValLabel)
% We use the parameters in our IEEE TPAMI submission
PCANet.NumStages = 2;
PCANet.PatchSize = 21;
PCANet.NumFilters = [50 20];
PCANet.HistBlockSize = [7 7];
PCANet.BlkOverLapRatio = 0.5;
fprintf('\n ====== PCANet Parameters ======= \n')
PCANet

%% PCANet Training with 10000 samples

fprintf('\n ====== PCANet Training ======= \n')
% TrnData_ImgCell = mat2imgcell(TrnData,ImgSize,ImgSize,ImgFormat); % convert columns in TrnData to cells 
% clear TrnData; 
tic;
[ftrain,V] = LDANet_train(TrnData,indian_pines_corrected,TrnLabels,Tr_idx_R,Tr_idx_C,PCANet,1); % BlkIdx serves the purpose of learning block-wise DR projection matrix; e.g., WPCA
PCANet_TrnTime = toc;
clear TrnData_ImgCell; 


fprintf('\n ====== Training Linear SVM Classifier ======= \n')
% for i=1:size(ftrain,2)
%     ftrain(:,i)=ftrain(:,i)/norm(ftrain(:,i));
% end
tic;
% models = train(TrnLabels', sparse(ftrain'), '-s 1 -q'); % we use linear SVM classifier (C = 1), calling libsvm library
LinearSVM_TrnTime = toc;



%% PCANet Feature Extraction and Testing 
fprintf('\n ====== PCANet Testing ======= \n')
TestData=zeros(200,size(Te_idx_C,2));
% for i=1:size(Te_idx_C)
%     TestData(:,i) = reshape(indian_pines_corrected(Te_idx_R(i),Te_idx_C(i),:),200,1); % 
% end
nCorrRecog = 0;
RecHistory = zeros(numel(Te_idx_C),1);

tic; 
ftest = LDANet_FeaExt_new(indian_pines_corrected,Te_idx_R,Te_idx_C,V,PCANet);
% for i=1:size(ftest,2)
%     ftest(:,i)=ftest(:,i)/norm(ftest(:,i));
% end
% for i=1:size(ftrain,2)
%     ftrain(:,i)=ftrain(:,i)/norm(ftrain(:,i));
% end

tic
[TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy(xxx),label_index_expected] = elm_kernel(ftrain, TrnLabels,ftest,TestLabels,1, 1000, 'RBF_kernel',45);
toc

[kappa(xxx), acc(:,xxx), acc_O(xxx), acc_A(xxx)] = evaluate_results(label_index_expected, TestLabels);
end
save Result_7.19_21train
% ==== PCANet Demo =======
clear all; close all; clc; 
addpath('./Utils');
addpath('./Liblinear');
% ImgFormat = 'gray'; %'color' or 'gray'
%% Loading data from MNIST Basic (10000 training, 2000 validation, 50000 testing) 
% load data
load('./Indian_pines/Indian_pines_corrected'); 
load('./Indian_pines/Indian_pines_gt'); 
% =======================================
Mi=min(min(min(indian_pines_corrected)));
Ma=max(max(max(indian_pines_corrected)));
for i=1:size(indian_pines_corrected,1)
    for j=1:size(indian_pines_corrected,2)
         indian_pines_corrected(i,j,:)=(indian_pines_corrected(i,j,:)-Mi)/(Ma-Mi);
% tempp=reshape(indian_pines_corrected(i,j,:),1,200);
% indian_pines_corrected(i,j,:)=indian_pines_corrected(i,j,:)/norm(tempp);
    end
end
[C D]=find(indian_pines_gt>0);

% HIdata=indian_pines_corrected(C,D,:);
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
 Num=ceil(numel(C)*0.1);
%     if numel(C)>50
%         Num=50;
%     else
%         Num=10;
%     end
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

% TestData = HIdata(:,TestLabels)';

% ===========================================================

nTestImg = length(TestLabels);

%% PCANet parameters (they should be funed based on validation set; i.e., ValData & ValLabel)
% We use the parameters in our IEEE TPAMI submission
PCANet.NumStages = 2;
PCANet.PatchSize = 25;
PCANet.NumFilters = [50 20];
PCANet.HistBlockSize = [7 7];
PCANet.BlkOverLapRatio = 0.5;
fprintf('\n ====== PCANet Parameters ======= \n')
PCANet

%% PCANet Training with 10000 samples

fprintf('\n ====== PCANet Training ======= \n')
% TrnData_ImgCell = mat2imgcell(TrnData,ImgSize,ImgSize,ImgFormat); % convert columns in TrnData to cells 
% clear TrnData; 
tic;
[ftrain,V] = LDANet_train(TrnData,indian_pines_corrected,TrnLabels,Tr_idx_R,Tr_idx_C,PCANet,1); % BlkIdx serves the purpose of learning block-wise DR projection matrix; e.g., WPCA
PCANet_TrnTime = toc;
clear TrnData_ImgCell; 


fprintf('\n ====== Training Linear SVM Classifier ======= \n')
% for i=1:size(ftrain,2)
%     ftrain(:,i)=ftrain(:,i)/norm(ftrain(:,i));
% end
tic;
% models = train(TrnLabels', sparse(ftrain'), '-s 1 -q'); % we use linear SVM classifier (C = 1), calling libsvm library
LinearSVM_TrnTime = toc;



%% PCANet Feature Extraction and Testing 
fprintf('\n ====== PCANet Testing ======= \n')
TestData=zeros(200,size(Te_idx_C,2));
% for i=1:size(Te_idx_C)
%     TestData(:,i) = reshape(indian_pines_corrected(Te_idx_R(i),Te_idx_C(i),:),200,1); % 
% end
nCorrRecog = 0;
RecHistory = zeros(numel(Te_idx_C),1);

tic; 
ftest = LDANet_FeaExt_new(indian_pines_corrected,Te_idx_R,Te_idx_C,V,PCANet);
% for i=1:size(ftest,2)
%     ftest(:,i)=ftest(:,i)/norm(ftest(:,i));
% end
% for i=1:size(ftrain,2)
%     ftrain(:,i)=ftrain(:,i)/norm(ftrain(:,i));
% end

tic
[TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy(xxx),label_index_expected] = elm_kernel(ftrain, TrnLabels,ftest,TestLabels,1, 1000, 'RBF_kernel',45);
toc

[kappa(xxx), acc(:,xxx), acc_O(xxx), acc_A(xxx)] = evaluate_results(label_index_expected, TestLabels);
end
save Result_7.19_25train

% ==== PCANet Demo =======
clear all; close all; clc; 
addpath('./Utils');
addpath('./Liblinear');
% ImgFormat = 'gray'; %'color' or 'gray'
%% Loading data from MNIST Basic (10000 training, 2000 validation, 50000 testing) 
% load data
load('./Indian_pines/Indian_pines_corrected'); 
load('./Indian_pines/Indian_pines_gt'); 
% =======================================
Mi=min(min(min(indian_pines_corrected)));
Ma=max(max(max(indian_pines_corrected)));
for i=1:size(indian_pines_corrected,1)
    for j=1:size(indian_pines_corrected,2)
         indian_pines_corrected(i,j,:)=(indian_pines_corrected(i,j,:)-Mi)/(Ma-Mi);
% tempp=reshape(indian_pines_corrected(i,j,:),1,200);
% indian_pines_corrected(i,j,:)=indian_pines_corrected(i,j,:)/norm(tempp);
    end
end
[C D]=find(indian_pines_gt>0);

% HIdata=indian_pines_corrected(C,D,:);
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
 Num=ceil(numel(C)*0.1);
%     if numel(C)>50
%         Num=50;
%     else
%         Num=10;
%     end
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

% TestData = HIdata(:,TestLabels)';

% ===========================================================

nTestImg = length(TestLabels);

%% PCANet parameters (they should be funed based on validation set; i.e., ValData & ValLabel)
% We use the parameters in our IEEE TPAMI submission
PCANet.NumStages = 2;
PCANet.PatchSize = 29;
PCANet.NumFilters = [50 20];
PCANet.HistBlockSize = [7 7];
PCANet.BlkOverLapRatio = 0.5;
fprintf('\n ====== PCANet Parameters ======= \n')
PCANet

%% PCANet Training with 10000 samples

fprintf('\n ====== PCANet Training ======= \n')
% TrnData_ImgCell = mat2imgcell(TrnData,ImgSize,ImgSize,ImgFormat); % convert columns in TrnData to cells 
% clear TrnData; 
tic;
[ftrain,V] = LDANet_train(TrnData,indian_pines_corrected,TrnLabels,Tr_idx_R,Tr_idx_C,PCANet,1); % BlkIdx serves the purpose of learning block-wise DR projection matrix; e.g., WPCA
PCANet_TrnTime = toc;
clear TrnData_ImgCell; 


fprintf('\n ====== Training Linear SVM Classifier ======= \n')
% for i=1:size(ftrain,2)
%     ftrain(:,i)=ftrain(:,i)/norm(ftrain(:,i));
% end
tic;
% models = train(TrnLabels', sparse(ftrain'), '-s 1 -q'); % we use linear SVM classifier (C = 1), calling libsvm library
LinearSVM_TrnTime = toc;



%% PCANet Feature Extraction and Testing 
fprintf('\n ====== PCANet Testing ======= \n')
TestData=zeros(200,size(Te_idx_C,2));
% for i=1:size(Te_idx_C)
%     TestData(:,i) = reshape(indian_pines_corrected(Te_idx_R(i),Te_idx_C(i),:),200,1); % 
% end
nCorrRecog = 0;
RecHistory = zeros(numel(Te_idx_C),1);

tic; 
ftest = LDANet_FeaExt_new(indian_pines_corrected,Te_idx_R,Te_idx_C,V,PCANet);
% for i=1:size(ftest,2)
%     ftest(:,i)=ftest(:,i)/norm(ftest(:,i));
% end
% for i=1:size(ftrain,2)
%     ftrain(:,i)=ftrain(:,i)/norm(ftrain(:,i));
% end

tic
[TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy(xxx),label_index_expected] = elm_kernel(ftrain, TrnLabels,ftest,TestLabels,1, 1000, 'RBF_kernel',45);
toc

[kappa(xxx), acc(:,xxx), acc_O(xxx), acc_A(xxx)] = evaluate_results(label_index_expected, TestLabels);
end
save Result_7.19_29train
% ==== PCANet Demo =======
clear all; close all; clc; 
addpath('./Utils');
addpath('./Liblinear');
% ImgFormat = 'gray'; %'color' or 'gray'
%% Loading data from MNIST Basic (10000 training, 2000 validation, 50000 testing) 
% load data
load('./Indian_pines/Indian_pines_corrected'); 
load('./Indian_pines/Indian_pines_gt'); 
% =======================================
Mi=min(min(min(indian_pines_corrected)));
Ma=max(max(max(indian_pines_corrected)));
for i=1:size(indian_pines_corrected,1)
    for j=1:size(indian_pines_corrected,2)
         indian_pines_corrected(i,j,:)=(indian_pines_corrected(i,j,:)-Mi)/(Ma-Mi);
% tempp=reshape(indian_pines_corrected(i,j,:),1,200);
% indian_pines_corrected(i,j,:)=indian_pines_corrected(i,j,:)/norm(tempp);
    end
end
[C D]=find(indian_pines_gt>0);

% HIdata=indian_pines_corrected(C,D,:);
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
 Num=ceil(numel(C)*0.1);
%     if numel(C)>50
%         Num=50;
%     else
%         Num=10;
%     end
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

% TestData = HIdata(:,TestLabels)';

% ===========================================================

nTestImg = length(TestLabels);

%% PCANet parameters (they should be funed based on validation set; i.e., ValData & ValLabel)
% We use the parameters in our IEEE TPAMI submission
PCANet.NumStages = 2;
PCANet.PatchSize = 33;
PCANet.NumFilters = [50 20];
PCANet.HistBlockSize = [7 7];
PCANet.BlkOverLapRatio = 0.5;
fprintf('\n ====== PCANet Parameters ======= \n')
PCANet

%% PCANet Training with 10000 samples

fprintf('\n ====== PCANet Training ======= \n')
% TrnData_ImgCell = mat2imgcell(TrnData,ImgSize,ImgSize,ImgFormat); % convert columns in TrnData to cells 
% clear TrnData; 
tic;
[ftrain,V] = LDANet_train(TrnData,indian_pines_corrected,TrnLabels,Tr_idx_R,Tr_idx_C,PCANet,1); % BlkIdx serves the purpose of learning block-wise DR projection matrix; e.g., WPCA
PCANet_TrnTime = toc;
clear TrnData_ImgCell; 


fprintf('\n ====== Training Linear SVM Classifier ======= \n')
% for i=1:size(ftrain,2)
%     ftrain(:,i)=ftrain(:,i)/norm(ftrain(:,i));
% end
tic;
% models = train(TrnLabels', sparse(ftrain'), '-s 1 -q'); % we use linear SVM classifier (C = 1), calling libsvm library
LinearSVM_TrnTime = toc;



%% PCANet Feature Extraction and Testing 
fprintf('\n ====== PCANet Testing ======= \n')
TestData=zeros(200,size(Te_idx_C,2));
% for i=1:size(Te_idx_C)
%     TestData(:,i) = reshape(indian_pines_corrected(Te_idx_R(i),Te_idx_C(i),:),200,1); % 
% end
nCorrRecog = 0;
RecHistory = zeros(numel(Te_idx_C),1);

tic; 
ftest = LDANet_FeaExt_new(indian_pines_corrected,Te_idx_R,Te_idx_C,V,PCANet);
% for i=1:size(ftest,2)
%     ftest(:,i)=ftest(:,i)/norm(ftest(:,i));
% end
% for i=1:size(ftrain,2)
%     ftrain(:,i)=ftrain(:,i)/norm(ftrain(:,i));
% end

tic
[TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy(xxx),label_index_expected] = elm_kernel(ftrain, TrnLabels,ftest,TestLabels,1, 1000, 'RBF_kernel',45);
toc

[kappa(xxx), acc(:,xxx), acc_O(xxx), acc_A(xxx)] = evaluate_results(label_index_expected, TestLabels);
end

save Result_7.19_33train





    