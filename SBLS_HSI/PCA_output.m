function OutImg = PCA_output(InImg,NumFilters, V)
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


% ImgZ = length(InImg);
% mag = (PatchSize-1)/2;
% % OutImg = cell(NumFilters,1); 
% cnt = 0;
% num=1;
    [ImgX, ImgY, NumChls] = size(InImg);
    [R C]=size(V);
    S=sqrt(R);
    for i=1:C
        F{i}=reshape(V(:,i),S,S);       
    end
    
    
OutImg=zeros(ImgX,ImgY,NumFilters*NumChls);
for k=1:NumChls
    
    temp=zeros(ImgX,ImgY,NumFilters);
    In= reshape(InImg(:,:,k),size(InImg,1),size(InImg,2));
     InImg1=padarray(In,[floor(S/2) floor(S/2)],'symmetric','both' );
    for i=1:NumFilters       
        temp(:,:,i)=filter2(F{i},InImg1,'valid');
    end
    OutImg(:,:,(k-1)*NumFilters+1:k*NumFilters)=temp;
end

   
