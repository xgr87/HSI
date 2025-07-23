function OutImg = PCA_output_new(InImg, NumFilters, V)
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

cnt = 0;
num=1;
    [ImgX, ImgY, NumChls] = size(InImg);
    
%     InImg1=zeros(ImgX+2*floor(PatchSize/2),ImgY+2*floor(PatchSize/2),NumChls);
% for k=1:NumChls
%     InImg1(:,:,k)=padarray(InImg(:,:,k),[floor(PatchSize/2) floor(PatchSize/2)],'symmetric','both' );    
% end
    
im=zeros(NumChls,ImgX*ImgY);
    for j=1:ImgY
        for i=1:ImgX
            temp=InImg(i,j,:);
        im(:,num)=reshape(temp,numel(temp),1);
        num=num+1;
        end
    end
   
    for j = 1:NumFilters
        cnt = cnt + 1;
        OutImg(:,:,cnt) = reshape(V(:,j)'*im,ImgX,ImgY);  % convolution output
    end
