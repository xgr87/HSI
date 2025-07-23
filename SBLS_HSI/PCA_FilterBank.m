function [V] = PCA_FilterBank(InImg,Tr_idx_R,Tr_idx_C,PatchSize, NumFilters) 
% =======INPUT=============
% InImg            Input images (cell structure)  
% InImgIdx         Image index for InImg (column vector)
% PatchSize        the patch size, asumed to an odd number.
% NumFilters       the number of PCA filters in the bank.
% givenV           the PCA filters are given. 
% =======OUTPUT============
% V                PCA filter banks, arranged in column-by-column manner
% OutImg           filter output (cell structure)
% OutImgIdx        Image index for OutImg (column vector)


addpath('./Utils')

% to efficiently cope with the large training samples, we randomly subsample 100000 training subset to learn PCA filter banks
ImgZ = length(InImg);
MaxSamples = 100000;
NumRSamples = min(ImgZ, MaxSamples); 
RandIdx = randperm(ImgZ);
RandIdx = RandIdx(1:NumRSamples);

%% Learning PCA filters (V

    im = im2col_general_new(InImg,Tr_idx_R,Tr_idx_C,[PatchSize PatchSize]); % collect all the patches of the ith image in a matrix
    im = im-mean(im,2)*ones(1,size(im,2)); % patch-mean removal
    Rx = im*im'; % sum of all the input images' covariance matrix

Rx = Rx/(size(im,2));
[E D] = eig(Rx);
[trash ind] = sort(diag(D),'descend');
V = E(:,ind(1:NumFilters));  % principal eigenvectors




 



