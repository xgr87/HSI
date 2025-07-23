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

NClass=16;
TrnLabels=[];
TestLabels=[];
Tr_idx_C=[];
Te_idx_C=[];
Te_idx_R=[];
Tr_idx_R=[];
train=zeros(145,145);
test=zeros(145,145);
for i=1:NClass   
    [R C]=find(indian_pines_gt==i);
    Num=ceil(numel(C)*0.2);
    idx_rand=randperm(numel(C));
    Tr_idx_C=C(idx_rand(1:Num))';
    Tr_idx_R=R(idx_rand(1:Num))';
    Te_idx_R=R(idx_rand(Num+1:end))';
    Te_idx_C=C(idx_rand(Num+1:end))';
    for k=1:Num
        train(Tr_idx_R(k),Tr_idx_C(k))=i;
    end
    
    for j=1:numel(idx_rand)-Num
        test(Te_idx_R(j),Te_idx_C(j))=i;
    end
end

test=uint8(test);
train=uint8(train);