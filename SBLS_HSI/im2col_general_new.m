function im = im2col_general_new(varargin)
% 

NumInput = length(varargin);
InImg = varargin{1};
patchsize12 = varargin{4}; 

z = size(InImg,3);
im = cell(z,1);
temp=zeros(patchsize12(1)^2,numel(varargin{2}));
for j=1:size(InImg,3)
    InImg1=padarray(InImg(:,:,j),[floor(patchsize12(1)/2) floor(patchsize12(1)/2)],'symmetric','both' );
for i=1:numel(varargin{2})
      temp(:,i)= im2colstep(InImg1(varargin{2}(i):varargin{2}(i)+patchsize12(1)-1,varargin{3}(i):varargin{3}(i)+patchsize12(1)-1),patchsize12);    
end
im{j}=temp;
end
% if NumInput == 2
%     for i = 1:z
%         im{i} = im2colstep(InImg(:,:,i),patchsize12)';
%     end
% else
%     for i = 1:z
%         im{i} = im2colstep(InImg(:,:,i),patchsize12,varargin{3})';
%     end 
% end
im = [im{:}];
    
    