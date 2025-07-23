function im = im2col_general_three(varargin)
% 

% NumInput = length(varargin);
InImg = varargin{1};
patchsize12 = varargin{4}; 

z = size(InImg,3);
im =zeros(patchsize12(1)^2*z,numel(varargin{2}));
temp=zeros(patchsize12(1)^2,numel(varargin{2}));
InImg1=zeros(size(InImg,1)+2*floor(patchsize12(1)/2),size(InImg,2)+2*floor(patchsize12(1)/2),z);
for j=1:size(InImg,3)
% for i=1:numel(varargin{2})
%       InImg1=padarray(InImg(:,:,j),[floor(patchsize12(1)/2) floor(patchsize12(1)/2)],'symmetric','both' );
%       temp(:,i)= im2colstep(InImg1(varargin{2}(i):varargin{2}(i)+patchsize12(1)-1,varargin{3}(i):varargin{3}(i)+patchsize12(1)-1),patchsize12);    
% end
InImg1(:,:,j)=padarray(InImg(:,:,j),[floor(patchsize12(1)/2) floor(patchsize12(1)/2)],'symmetric','both' );

% im{j}=temp;
end
for i=1:numel(varargin{2})
     xx=InImg1(varargin{2}(i):varargin{2}(i)+patchsize12(1)-1,varargin{3}(i):varargin{3}(i)+patchsize12(1)-1,:);
     im(:,i)= reshape(xx,numel(xx),1);    
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
% im = [im{:}];
    
    