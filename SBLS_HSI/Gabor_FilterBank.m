function [fSiz,V] = Gabor_FilterBank(InImg,Tr_idx_R,Tr_idx_C,PatchSize, NumFilters) 
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
rot = [90 -75 -60 -45 -30 -15 0 15 30 45 60 75];
c1ScaleSS = [10];
RF_siz    = [25];
c1SpaceSS = [14];
minFS     = 15;
maxFS     = 15;
div = [3.55];
Div       = div;
%--- END Settings for Testing --------%

fprintf(1,'Initializing gabor filters -- full set...');
%creates the gabor filters use to extract the S1 layer
[fSiz,V] = init_gabor(rot,RF_siz,Div);




 



