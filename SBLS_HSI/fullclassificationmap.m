% ==== PCANet Demo =======
clear all; close all; clc; 
% addpath('./Utils');
% addpath('./Liblinear');
%% 
% load data
% load('./Indian_pines/Indian_pines_corrected'); 
% load('./Indian_pines/Indian_pines_gt'); 
load('Indian_pines_color.mat')

TestLabels=ones(1,145*145); 

ftest=zeros(size(f,3),numel(145*145));
i=1;
for kkk=1:145
    for ttt=1:145
    ftest(:,i)=reshape(f(ttt,kkk,:),size(f,3),1);
    i=i+1;
    end
end

pa1=1000;
pa2=0.1

label_index_expected=[];
tic
[TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy(xxx),label_index_expected] = elm_kernel(ftrain, TrnLabels,ftest,TestLabels,1, pa1, 'RBF_kernel',pa2);
toc

M=reshape(label_index_expected,145,145);


colormap(1,:)=[];
imshow(reshape(label_index_expected,145,145),colormap,'Border','tight');




