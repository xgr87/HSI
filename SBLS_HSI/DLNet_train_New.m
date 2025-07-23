function [OutImg, V,fSiz] = DLNet_train_New(InImg,TrnLabels,Tr_idx_R,Tr_idx_C,DLNet)
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

for i=1:NumStage
if(rem(i,2))
display(['Computing LDA filter bank and its outputs at stage ' num2str(i) '...'])
V{i} = LDA_FilterBank(OutImg, TrnLabels,Tr_idx_R,Tr_idx_C); % compute PCA filter banks
OutImg = LDA_output(OutImg, DLNet.NumFilters(i), V{i});
else
display(['Computing PCA filter bank and its outputs at stage ' num2str(i) '...'])
    
[fSiz,V{i}]= Gabor_FilterBank(OutImg,Tr_idx_R,Tr_idx_C,DLNet.PatchSize(i), DLNet.NumFilters(i)); % compute PCA filter banks


OutImg = Gabor_output(OutImg,DLNet.NumFilters(i), V{i},fSiz);  % compute the last PCA outputs of image "idx"
end
end









