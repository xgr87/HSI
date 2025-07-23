%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [labels_t,numlabels]=SuperpixelSegmentation(data,numSuperpixels)
% RCSPP ³¬ÏñËØ·Ö¸î
 
[row, col, dim] = size(data);
input_img = zeros(1, row * col * dim);
startpos = 1;
for i = 1 : row
    for j = 1 : col
        input_img(startpos : startpos + dim - 1) = data(i, j, :);
        startpos = startpos + dim;
    end
end


%% perform Regional Clustering

%numSuperpixels = 200;  % number of segments
compactness = 0.1; % compactness2 = 1-compactness, compactness*dxy+compactness2*dspectral
dist_type = 2; % 1:ED£»2£ºSAD; 3:SID; 4:SAD-SID
seg_all = 1; % 1: All pixels are clustered£¬ 2£ºexist un-clustered pixels
% labels:segment no of each pixel
% numlabels: actual number of segments
[labels, numlabels, ~, ~] = RCSPP(input_img, row, col, dim, numSuperpixels, compactness, dist_type, seg_all);
clear input_img;

labels_t = zeros(row, col, 'int32');
for i=1:row
    for j=1:col
        labels_t(i,j) = labels((i-1)*col+j);
    end
end
end