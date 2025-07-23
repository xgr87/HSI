function [f, V] = LDANet_train(TrnData,InImg,TrnLabels,Tr_idx_R,Tr_idx_C,PCANet,IdtExt)
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

if length(PCANet.NumFilters)~= PCANet.NumStages;
    display('Length(PCANet.NumFilters)~=PCANet.NumStages')
    return
end

NumImg = numel(Tr_idx_R);

V = cell(PCANet.NumStages,1); 
OutImg = InImg; 
ImgIdx = (1:NumImg)';
clear InImg; 


display(['Computing PCA filter bank and its outputs at stage ' num2str(1) '...'])
V{1} = LDA_FilterBank_First(OutImg, TrnLabels,Tr_idx_R,Tr_idx_C,PCANet.PatchSize(1)); % compute PCA filter banks
[OutImg,ImgIdx] = LDA_output(OutImg, ImgIdx,Tr_idx_R,Tr_idx_C, PCANet.PatchSize(1), PCANet.NumFilters(1), V{1});

display(['Computing PCA filter bank and its outputs at stage ' num2str(2) '...'])
    
[V{2}]= PCA_FilterBank(OutImg,Tr_idx_R,Tr_idx_C,PCANet.PatchSize(2), PCANet.NumFilters(2)); % compute PCA filter banks
%   if stage ~= PCANet.NumStages % compute the PCA outputs only when it is NOT the last stage
%      [OutImg ImgIdx] = PCA_output(OutImg, ImgIdx, PCANet.PatchSize, PCANet.NumFilters(stage), V{2});  
%   end



if IdtExt == 1 % enable feature extraction

        
        
        OutImg_i = PCA_output_new(OutImg,Tr_idx_R,Tr_idx_C,PCANet.PatchSize(2), PCANet.NumFilters(end), V{end});  % compute the last PCA outputs of image "idx"
        
%         f = HashingHist_new(PCANet,OutImg_i); % compute the feature of image "idx"
       f=OutImg_i;
end







