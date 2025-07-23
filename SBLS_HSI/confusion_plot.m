%  Matlab package for statistics and visualization of classification results and many other problems.
%  
%  Author£º Page( Ø§×Ó)
%           Blog: www.shamoxia.com;
%           QQ:379115886;
%           Email: peegeelee@gmail.com
%  Date:    Dec. 2010
%% initialization
% clc;
% clear;
% load num_in_class;    % instance number of each class
% load actual_label;    % actual label of each instance
% load predict_label;   % predicted label of your experiments
% load decision_values; % deccision values of each instance in your classification experiments(e.g. dec_values of Libsvm)
% load name_class;      % name of each class

%% compute and visualize the confusion matrix
clear
% addpath PG_Curve; % package for computing confusion matrix
% addpath ConfusionMatrices

confusion_matrix=zeros(16,16);
predict_label=[];
name_class{1}='Alfalfa';
name_class{2}='Corn-notill';
name_class{3}='Corn-mintill';
name_class{4}='Corn';
name_class{5}='Grass-pasture';
name_class{6}='Grass-trees';
name_class{7}='Grass-pasture-mowed';
name_class{8}='Hay-windrowed';
name_class{9}='Oats';
name_class{10}='Soybean-notill';
name_class{11}='Soybean-mintill';
name_class{12}='Soybean-clean';
name_class{13}='Wheat';
name_class{14}='Woods';
name_class{15}='Buildings-Grass-Trees-Drives';
name_class{16}='Stone-Steel-Towers';



% for mm=1:10
%     j=num2str(mm);
%     s=['Result_macau_large_new_',j,'.mat'];
    load ('Result_MFA_83_7_reg0.15_01.mat')
    
clabel = unique(TestLabels);
nclass = length(clabel);
num_in_class=[];
nRounds=1;
accuracy = zeros(nRounds, 1);
class_name=(1:nclass);
 predict_label=[];
 confusion_matrix=zeros(16,16);
 for kk=1:10
     curr_pred_label=[];
     num_in_class=[];
      predict_label=[];
    for jj = 1 : length(class_name),
        c = class_name(jj);
        idx = find(TestLabels == c);
        curr_pred_label = label_index_expected(kk,idx);
%         curr_gnd_label = label_index_expected(idx); 
        num_in_class=[num_in_class numel(idx)];
        predict_label=[predict_label curr_pred_label];
    end; 
    
%     accuracy_mean(mm) = mean(acc); 
    confusion_matrix=confusion_matrix+compute_confusion_matrix(predict_label',num_in_class',name_class);    
 end

draw_cm(confusion_matrix*100/10,name_class,16);



