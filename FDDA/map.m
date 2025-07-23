% clc,clear
% close all
% 
% load('Indian_pines_gt.mat');
% Label = indian_pines_gt;

% load('Salinas_gt.mat');
% Label = salinas_gt;

% load('PaviaU_gt.mat');
% Label = paviaU_gt;
k = 18;
% k = 18;
A = double(Alllabel)/k;
A(~A) = 1;
% 
% B = double(pixellabel)/k;
% B(~B) = 1;

% C = double(boxlabel)/k;
% C(~C) = 1;

% D = double(Label)/k;
% D(~D) = 1;

imshow(A), colormap(colorcube), axis image;
% imshow(B), colormap(colorcube), axis image;
% imshow(C), colormap(colorcube), axis image;
% imshow(D), colormap(colorcube), axis image;
iptsetpref('ImshowBorder','tight')
iptsetpref('ImtoolInitialMagnification','fit')

%% »­µ÷É«¿é
% num = 9;  %the number of classes
% n = 50;  % the size of the block for each class
% cbarblock = ones(n,n);
% spaceline = 20;
% block = zeros(num*n+(num-1)*spaceline,n);
% for i = 1 : num
%     block((n+spaceline)*(i-1)+1:(n+spaceline)*(i-1)+n,:) = cbarblock*i/k;
% end
% block(~block) = 1;
% imshow(block), colormap(colorcube),
% iptsetpref('ImshowBorder','tight')
% iptsetpref('ImtoolInitialMagnification','fit')