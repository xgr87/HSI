clc
clear
close all
Results = zeros(4,10);
for step = 8:10
    tic;
run trainsetselect;

%%%% X. Kang, et al,PCA-Based Edge-Preserving Features for Hyperspectral Image Classification
addpath ('functions')
addpath (genpath('libsvm-3.22'))
%% load original image
% path='.\PCA-EPFs\';
% inputs = 'IndiaP';%145*145*200/10249/16
% inputs = 'Inblock_P';
% inputs = 'SalinasP';
% inputs = 'PaviaUP';
% location = [path,inputs];
% load (location);
% load ('IndiaP.mat');
%%% size of image
[no_lines, no_rows, no_bands] = size(img);
GroundT=GroundT';
% load ('in_1.mat');
% load (['.\training_indexes\Inblock_1.mat'])
% load (['.\training_indexes\Sa_1.mat'])
% load (['.\training_indexes\PU_1.mat'])
%% Spectral dimension Reduction
img2=average_fusion(img,10);
OA=[];AA=[];kappa=[];CA=[];
for i=1:10
    indexes=XX(:,i);
    %% Normalization
    no_bands=size(img2,3);
    fimg=reshape(img2,[no_lines*no_rows no_bands]);
    [fimg] = scale_new(fimg);
    fimg=reshape(fimg,[no_lines no_rows no_bands]);
    %% Feature extraction
     fimg1=spatial_feature(fimg,30,0.3);
     fimg2=spatial_feature(fimg,115,0.6);
     fimg3=spatial_feature(fimg,200,0.9);
     f_fimg=cat(3,fimg1,fimg2,fimg3);
    %% Feature fusion with the PCA
    fimg=PCA_img(f_fimg, 'first');
%     fimg=PCA_img(f_fimg, 30);
    %% SVM classification
    fimg = ToVector(fimg);
    fimg = fimg';
    fimg=double(fimg);
    %%%
    train_SL = GroundT(:,indexes);
    train_samples = fimg(:,train_SL(1,:))';
    train_labels= train_SL(2,:)';
    %%%
    test_SL = GroundT;
    test_SL(:,indexes) = [];
    test_samples = fimg(:,test_SL(1,:))';
    test_labels = test_SL(2,:)';
    % Normalizing Training and original img
    [train_samples,M,m] = scale_func(train_samples);
    [fimg ] = scale_func(fimg',M,m);
    % Selecting the paramter for SVM
    [Ccv, Gcv, cv, cv_t]=cross_validation_svm(train_labels,train_samples);
%     Ccv = 1000; Gcv =0.1;
    % Training using a Gaussian RBF kernel
    % parameter=sprintf('-c %f -g %f -m 500 -t 2 -q',Ccv,Gcv);
    parameter=sprintf('-c %f -g %f -t 2',Ccv,Gcv);
    model=svmtrain(train_labels,train_samples,parameter);
    % Testing
    Result = svmpredict(ones(no_lines*no_rows,1),fimg,model); %%%
    % Evaluation
    GroudTest = double(test_labels(:,1));
    ResultTest = Result(test_SL(1,:),:);
    [OA_i,AA_i,kappa_i,CA_i]=confusion(GroudTest,ResultTest);
    OA=[OA OA_i];
    AA=[AA AA_i];
    kappa=[kappa kappa_i];
%     CA=[CA CA_i];
%     Result = reshape(Result,no_lines,no_rows);
%     label_PCAEPFs(:,:,i) = Result;
%     VClassMap=label2colord(Result,'india');
%     figure,imshow(VClassMap);
end
% OA_std=std(OA);OA1=mean(OA);
% AA_std=std(AA);AA1=mean(AA);
% K_std=std(kappa);KA1=mean(kappa);
% accurperclass = CA;

Results(:,step) = [mean(OA);mean(AA);mean(kappa);step];
clear OA AA kappa
toc;
end

% save honghu_PCAEPFs.mat OA AA kappa accurperclass label_PCAEPFs;