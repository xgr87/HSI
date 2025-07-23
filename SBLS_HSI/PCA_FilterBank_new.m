function V = PCA_FilterBank_new(InImg,TrnLabels,Tr_idx_R,Tr_idx_C) 
% =======INPUT=============
% InImg            Input images (cell structure)  
% InImgIdx         Image index for InImg (column vector)
% PatchSize        the patch size, asumed to an odd number.
% NumFilters       the number of PCA filters in the bank.
% givenV           the PCA filters are given. 
addpath('./Utils')
im=zeros(size(InImg,3),numel(Tr_idx_R));
for i=1:numel(Tr_idx_R)
    im(:,i)=InImg(Tr_idx_R(i),Tr_idx_C(i),:);
end



options=[];
options.ReducedDim=150;
% options.PCARatio=140;
[V, eigvalue] = PCA(im',options);
% im=im-mean(im,2)*ones(1,size(im,2));
% % im = bsxfun(@minus, im', mean(im')); % patch-mean removal 
% Rx = Rx + im*im'; % sum of all the input images' covariance matrix
% Rx = Rx/(size(im,2));
% [E D] = eig(Rx);
% [trash ind] = sort(diag(D),'descend');
% V = E(:,ind(1:NumFilters));  % principal eigenvectors

%%
