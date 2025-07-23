% clc,clear

dir = '..\datasets\';
addpath(genpath(pwd))

load([dir,'PaviaU.mat']);
load([dir,'PaviaU_gt.mat']);
spectral = paviaU(:,:,1:step:end);
[n,m,dim] = size(spectral);
Xlabel = double(reshape(paviaU_gt,n*m,1));
img = double(spectral);

% load([dir,'WHU_Hi_HongHu.mat']);
% load([dir,'WHU_Hi_HongHu_gt.mat']);
% spectral = WHU_Hi_HongHu;
% [n,m,dim] = size(spectral);
% Xlabel = double(reshape(WHU_Hi_HongHu_gt,n*m,1));
% img = double(spectral);
%% 读取背景像素
XX = [];
GroundTind = [];
numberoflabel = max(Xlabel);
for i = 1:numberoflabel
    ind = find(Xlabel==i);
    GroundTind = [GroundTind;ind];
    clear ind;
end
GroundT = [GroundTind, Xlabel(GroundTind)];

for randi = 1 : 10
    %% 选取样本
    ratio = 0.1;   % 训练样本比例
%     tationum = 5;  % 训练样本个数/每类
    traindataNo=[];testdataNo=[];
    numberofdata = zeros(1,numberoflabel);
    Labelmatrix = zeros(n,m);
    for i = 1:numberoflabel
        
        ind = find(Xlabel==i);
        numberofdata(i) = length(ind);
        if(numberofdata(i) ~= 0)
            No = randperm(numberofdata(i));
            Numper = ceil(numberofdata(i)*ratio);
%             Numper = tationum;
            traindataNo = [traindataNo;ind(No(1:Numper))];
        end
    end
    for  i = 1 :  length(traindataNo)
        XX(i,randi) = find(GroundT(:,1) == traindataNo(i));
    end
end
% save in_1.mat XX;
% save IndiaP.mat GroundT img