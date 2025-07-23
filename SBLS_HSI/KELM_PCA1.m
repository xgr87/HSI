% ==== PCANet Demo =======
clear all; close all; clc; 
addpath('./Utils');
addpath('./Liblinear');
Start = clock;
%% 
% load data
load ./Indian_pines/Indian_pines_corrected
load ./Indian_pines/Indian_pines_gt

indian_pines_corrected_norm = rescale(indian_pines_corrected);

L_grain1 = 100;
indian_pines_corrected1=[];
for i=1:size(indian_pines_corrected_norm,1)
    for j=1:size(indian_pines_corrected_norm,2)
         ttt = reshape(indian_pines_corrected_norm(i,j,:),1,200);
         [c,l]=wavedec(ttt,3,'db5');
%          a3=wrcoef('a',c,l,'db5',3);
         a2=wrcoef('a',c,l,'db5',2);
         a1=wrcoef('a',c,l,'db5',1);
         aa=a2(2:end)-a2(1:end-1);
%          aa1=aa(2:end)-aa(1:end-1);
         
         ga1=[];
         ga2=[];
         gaa=[];
         for k=1:size(a1,2)-L_grain1+1
             ga1=[ga1,a1(1,k:k+L_grain1-1)];
             ga2=[ga2,a2(1,k:k+L_grain1-1)];
             if k<=size(aa,2)-L_grain1+1
                 gaa=[gaa,aa(1,k:k+L_grain1-1)];
             end
         end
         indian_pines_corrected1(i,j,:)=[ga1 ga2 gaa];
    end
end

indian_pines_corrected = indian_pines_corrected1;

[C, ~]=find(indian_pines_gt>0);

TestingAccuracy=zeros(1,10);
TestingAccuracy_o=zeros(1,10);
tt=[];
for xxx=1:1
NClass=16;
TrnLabels=[];
TestLabels=[];
Tr_idx_C=[];
Te_idx_C=[];
Te_idx_R=[];
Tr_idx_R=[];
for i=1:NClass   
    [R C]=find(indian_pines_gt==i);
    Num=ceil(numel(C)*0.02);
    idx_rand=randperm(numel(C));
    Tr_idx_C=[Tr_idx_C C(idx_rand(1:Num))'];
    Tr_idx_R=[Tr_idx_R R(idx_rand(1:Num))'];
    Te_idx_R=[Te_idx_R R(idx_rand(Num+1:end))'];
    Te_idx_C=[Te_idx_C C(idx_rand(Num+1:end))'];
    TrnLabels=[TrnLabels ones(1,Num)*i];
    TestLabels=[TestLabels ones(1,numel(C)-Num)*i]; 
end

temp_Tr=zeros(16, numel(TrnLabels));
for i = 1:numel(TrnLabels)
    temp_Tr(TrnLabels(i),i)=1;
end
Tr=temp_Tr*2-1;

temp_Te=zeros(16, numel(TestLabels));
for i = 1:numel( TestLabels)
    temp_Te(TestLabels(i),i)=1;
end
Te=temp_Te*2-1;


% ===========================================================
nTestImg = length(TestLabels);
%%
% We use the parameters in our
DLNet.NumStages = 8;
DLNet.PatchSize = [1 21 1 21 1 21 1 21];
DLNet.NumFilters = [16 6 16 6 16 6 16 6];
%DLNet.HistBlockSize = [7 7];
fprintf('\n ====== DLNet Parameters ======= \n')
DLNet

%% PCANet Training with 10000 samples

fprintf('\n ======DLNet Training ======= \n')

[f,V] = DLNet_train(indian_pines_corrected,TrnLabels,Tr_idx_R,Tr_idx_C,DLNet,Te_idx_R,Te_idx_C,TestLabels); % 
ftrain=zeros(size(f,3),numel(Tr_idx_R));
for i=1:numel(Tr_idx_R)
    ftrain(:,i)=reshape(f(Tr_idx_R(i),Tr_idx_C(i),:),size(f,3),1);
end

clear TrnData_ImgCell; 

fprintf('\n ====== Training Classifier ======= \n')

%% PCANet Feature Extraction and Testing 
fprintf('\n ====== MFANet Testing ======= \n')
TestData=zeros(200,size(Te_idx_C,2));

nCorrRecog = 0;
RecHistory = zeros(numel(Te_idx_C),1);

ftest=zeros(size(f,3),numel(Te_idx_R));
for i=1:numel(Te_idx_R)
    ftest(:,i)=reshape(f(Te_idx_R(i),Te_idx_C(i),:),size(f,3),1);
end

% pa1=1000;
% pa2=0.1;
C = 2^-30; s = .5;%the l2 regularization parameter and the shrinkage scale of the enhancement nodes
N11=15;%feature nodes  per window
N2=20;% number of windows of feature nodes
N33=6000;% number of enhancement nodes
N1=N11; N3=N33;  
% [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy(xxx),label_index_expected(xxx,:)] = elm_kernel(ftrain, TrnLabels,ftest,TestLabels,1, pa1, 'RBF_kernel',pa2);
[TrainingAccuracy,TestingAccuracy,Training_time,Testing_time,label_index_expected(xxx,:)] ...
    = bls_train(ftrain',Tr',ftest',Te',s,C,N1,N2,N3);  

% [kappa(xxx), acc(:,xxx), acc_O(xxx), acc_A(xxx)] = evaluate_results(label_index_expected(xxx,:), TestLabels);


% load('Indian_pines_color.mat')
% figure
% TestLabels=ones(1,145*145); 
% 
% ftest=zeros(size(f,3),numel(145*145));
% i=1;
% for kkk=1:145
%     for ttt=1:145
%     ftest(:,i)=reshape(f(ttt,kkk,:),size(f,3),1);
%     i=i+1;
%     end
% end
% 
% % pa1=1000;
% % pa2=0.1
% temp_Te=zeros(16, numel(TestLabels));
%     for i = 1:numel(TestLabels)
%         temp_Te(TestLabels(i),i)=1;
%     end
%  Te=temp_Te*2-1;
% % label_index_expected=[];
% 
% % [TrainingTime1, TestingTime1, TrainingAccuracy1, TestingAccuracy1,label_index_expected1] = elm_kernel(ftrain, TrnLabels,ftest,TestLabels,1, pa1, 'RBF_kernel',pa2);
% [TrainingAccuracy,TestingAccuracy,Training_time,Testing_time,label_index_expected1] = bls_train(ftrain',Tr',ftest',Te',s,C,N1,N2,N3);  
% 
% M=reshape(label_index_expected1,145,145);
% 
% 
% colormap(1,:)=[];
% imshow(reshape(label_index_expected1,145,145),colormap,'Border','tight');

end

End = clock;
time = etime(End,Start)

% save result_20180824_P_4




    


    