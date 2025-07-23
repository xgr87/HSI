function [InImg, V] = DLNet_train_M(InImg,TrnLabels,Tr_idx_R,Tr_idx_C,DLNet,Te_idx_R,Te_idx_C,TestLabels)
% =======INPUT=============
% InImg     Input images (cell); each cell can be either a matrix (Gray) or a 3D tensor (RGB)  
% PCANet    PCANet parameters (struct)
%       .PCANet.NumStages      
%           the number of stages in PCANet; e.g., 2  
%       .PatchSize
%           the patch size (filter size) for square patches; e.g., 3, 5, 7
%           only a odd number allowed
%       .NumFilters
%           the number of filters in each stage; e.g., [16 8] means 16 and
%           8 filters in the first stage and second stage, respectively
%       .HistBlockSize 
%           the size of each block for local histogram; e.g., [10 10]
%       .BlkOverLapRatio 
%           overlapped block region ratio; e.g., 0 means no overlapped 
%           between blocks, and 0.3 means 30% of blocksize is overlapped 
% IdtExt    a number in {0,1}; 1 do feature extraction, and 0 otherwise  
% =======OUTPUT============
% f         PCANet features (each column corresponds to feature of each image)
% V         learned PCA filter banks (cell)
% BlkIdx    index of local block from which the histogram is compuated
% ========= CITATION ============
% T.-H. Chan, K. Jia, S. Gao, J. Lu, Z. Zeng, and Y. Ma, 
% "PCANet: A simple deep learning baseline for image classification?" submitted to IEEE TPAMI. 
% ArXiv eprint: http://arxiv.org/abs/1404.3606 

% Tsung-Han Chan [thchan@ieee.org]
% Please email me if you find bugs, or have suggestions or questions!

addpath('./Utils')

if length(DLNet.NumFilters)~= DLNet.NumStages;
    display('Length(DLNet.NumFilters)~=DLNet.NumStages')
    return
end

NumImg = numel(Tr_idx_R);

V = cell(DLNet.NumStages,1); 
ImgIdx = (1:NumImg)';

OutImg=InImg;
NumStage=DLNet.NumStages;
options = []; 

      options.intraK =1000; 
      options.interK = 1000; 
      options.Regu = 1; 
% options.Fisherface=1;
% options.keepMean=1;
      options.Dim=100;
for i=1:NumStage
if(rem(i,2))
    
    display(['Computing PCA filter bank and its outputs at stage ' num2str(i) '...'])
    
[V{i}]= PCA_FilterBank(InImg,Tr_idx_R,Tr_idx_C,DLNet.PatchSize(i), DLNet.NumFilters(i)); % compute PCA filter banks

InImg = PCA_output(InImg,DLNet.NumFilters(i), V{i});  % compute the last PCA outputs of image "idx"
InImg= F_output(InImg,3);

% V{i} = MFA_FilterBank(OutImg, TrnLabels,Tr_idx_R,Tr_idx_C,options); % compute PCA filter banks
% OutImg = MFA_output(OutImg, DLNet.NumFilters(i), V{i});
else
display(['Computing filter bank and its outputs at stage ' num2str(i) '...'])
pa1=[600 100000 100000];
pa2=[0.5 10 10];

[r1 c1 d1]=size(InImg);
ftest=[];
ftrain=[];
for i1=1:numel(Tr_idx_R)
    ftrain(:,i1)=reshape(InImg(Tr_idx_R(i1),Tr_idx_C(i1),:),size(InImg,3),1);
end


for i1=1:r1
    for j1=1:c1
        ftest=[ftest reshape(InImg(i1,j1,:),size(InImg,3),1)];
    end
end

TestLabels1=ones(1,r1*c1);

T = elm_kernel_new(ftrain,TrnLabels,ftest,TestLabels1,1, pa1(fix(i/2)), 'RBF_kernel',pa2(fix(i/2)));
tt=1;
InImg=zeros(r1,c1,size(T,1));
for i1=1:r1
    for j1=1:c1
        InImg(i1,j1,:)=T(:,tt);
        tt=tt+1;
    end
end
[xx mmmmm]=max(T);
tt=1;
tlab=zeros(r1,c1);
for i1=1:r1
    for j1=1:c1
        tlab(i1,j1)=mmmmm(tt);
        tt=tt+1;
    end
end
numx=0;

for i1=1:numel(Te_idx_R)
    if(tlab(Te_idx_R(i1),Te_idx_C(i1))==TestLabels(i1));
        numx=numx+1;
    end
end

turetex=numx/numel(Te_idx_R)
end
end









