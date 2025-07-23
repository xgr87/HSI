% This is the demo for the HBLS models

%% load the dataset，‘DIP’ or 'UH'
clear;
warning off all;
format compact;

load('./Indian_pines/Indian_pines_corrected');
load('./Indian_pines/Indian_pines_gt');

samples = double(reshape(indian_pines_corrected, 145*145, 200));
labels = double(reshape(indian_pines_gt, 145*145, 1));

Min=min(min(samples));
Max=max(max(samples));
samples = (samples-Min)/(Max-Min);
N=find(labels>0);

repeat = 10; % 重复实验次数
test_acc=zeros(1,repeat);
for pos = 1:repeat
    NClass=16;
    x_train=[];
    y_train=[];
    x_test=[];
    y_test=[];
    
    for i=1:NClass
        C=find(labels==i);
        Num=ceil(numel(C)*0.02);
        idx_rand=randperm(numel(C));
        x_train=[x_train; samples(C(idx_rand(1:Num)),:)];
        y_train=[y_train; i*ones(Num,1)];
        x_test=[x_test; samples(C(idx_rand(Num+1:end)),:)];
        y_test=[y_test; i*ones(numel(C)-Num,1)];
    end
    
    %% the samples are preprocessed and the lable data train_y and test_y are reset as N*C matrices
    model = svmtrain(y_train, x_train, '-c 100 -g 0.5');
    [predict_label, accuracy, preb] = svmpredict(y_test, x_test, model);
    test_acc(pos)=accuracy(1);
end


