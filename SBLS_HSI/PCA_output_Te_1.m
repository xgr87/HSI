function [OutImg,OutImgIdx] = PCA_output_Te_1(OutImg, Te_idx_R, Te_idx_C,PatchSize,NumFilters,V)
% Computing PCA filter outputs
% ======== INPUT ============
% InImg         Input images (cell structure); each cell can be either a matrix (Gray) or a 3D tensor (RGB)   
% InImgIdx      Image index for InImg (column vector)
% PatchSize     Patch size (or filter size); the patch is set to be sqaure
% NumFilters    Number of filters at the stage right before the output layer 
% V             PCA filter banks (cell structure); V{i} for filter bank in the ith stage  
% ======== OUTPUT ===========
% OutImg           filter output (cell structure)
% OutImgIdx        Image index for OutImg (column vector)
% ========= CITATION ============
% T.-H. Chan, K. Jia, S. Gao, J. Lu, Z. Zeng, and Y. Ma, 
% "PCANet: A simple deep learning baseline for image classification?" submitted to IEEE TPAMI. 
% ArXiv eprint: http://arxiv.org/abs/1404.3606 

% Tsung-Han Chan [thchan@ieee.org]
% Please email me if you find bugs, or have suggestions or questions!

addpath('./Utils')
InImg=OutImg;
% ImgZ = length(InImg);
% mag = (PatchSize-1)/2;
% OutImg = cell(NumFilters,1); 
cnt = 0;
num=1;
    [ImgX, ImgY, NumChls] = size(InImg);
    

    
V1=V{1};

  InImg1=zeros(ImgX+2*floor(PatchSize(1)/2),ImgY+2*floor(PatchSize(1)/2),NumChls);
for k=1:NumChls
    InImg1(:,:,k)=padarray(OutImg(:,:,k),[floor(PatchSize(1)/2) floor(PatchSize(1)/2)],'symmetric','both' );    
end
    
im=zeros(NumChls*PatchSize(1)^2,ImgX*ImgY);
    for j=1:ImgY
        for i=1:ImgX
            temp=InImg1(i:i+PatchSize(1)-1,j:j+PatchSize(1)-1,:);
        im(:,num)=reshape(temp,numel(temp),1);
        num=num+1;
        end
    end
   
    for j = 1:NumFilters
        cnt = cnt + 1;
        OutImg(:,:,cnt) = reshape(V1(:,j)'*im,ImgX,ImgY);  % convolution output
    end


for k=1:NumFilters(1)
    InImg1=padarray(OutImg(:,:,k),[floor(PatchSize(2)/2) floor(PatchSize(2)/2)],'symmetric','both' );
    num=1;
for i=1:numel(Te_idx_R)
    im1(:,num)=reshape(InImg1(Te_idx_R(i):Te_idx_R(i)+PatchSize(2)-1,Te_idx_C(i):Te_idx_C(i)+PatchSize(2)-1),PatchSize(2)^2,1);
    num=num+1;
end
    im2{k}=im1;
end
cnt=0;
 OutImg=[];
 V2=V{2};
 for j = 1:NumFilters(1)
        OutImg =[OutImg;V2'*im2{j}]; % convolution output
 end




