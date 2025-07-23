function OutImg = MFANet_Test(InImg,V,DLNet)
% =======INPUT=============
% InImg     Input images (cell)  
% V         given PCA filter banks (cell)
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
% =======OUTPUT============
% f         PCANet features (each column corresponds to feature of each image)
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

cnt = 0;
num=1;
NumFilters=DLNet.NumFilters;
   
[ImgX, ImgY, NumChls] = size(InImg);
 
OutImg=InImg;
NumStage=DLNet.NumStages;
for i=1:NumStage
    V1=V{i};
if(rem(i,2))
  OutImg = MFA_output(OutImg,DLNet.NumFilters(i), V1);
else
  OutImg= PCA_output(OutImg,NumFilters(i), V1);
OutImg= F_output(OutImg,3);
end
end




     




% for stage = 1:PCANet.NumStages
%      [OutImg ImgIdx] = PCA_output(OutImg, ImgIdx, ...
%            PCANet.PatchSize, PCANet.NumFilters(stage), V{stage});  
% end
% 
% [f BlkIdx] = HashingHist_new(PCANet,ImgIdx,OutImg);





