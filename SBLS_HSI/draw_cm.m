function draw_cm(mat,tick,num_class)
%%
%  Matlab code for visualization of confusion matrix;
%  Parameters��mat: confusion matrix;
%              tick: name of each class, e.g. 'class_1' 'class_2'...
%              num_class: number of class
%
%  Author�� Page( ا��)  
%           Blog: www.shamoxia.com;  
%           QQ:379115886;  
%           Email: peegeelee@gmail.com
%%
iptsetpref('ImshowBorder','tight')
iptsetpref('ImtoolInitialMagnification','fit')
set(gcf,'Position',[400,100,850,750])
imagesc(mat);            %# in color
% colormap(flipud(cool));  %# for gray; black for large value.
colormap((cool));  %# for gray; black for large value.
textStrings = num2str(mat(:),'%0.2f');  
textStrings = strtrim(cellstr(textStrings)); 
[x,y] = meshgrid(1:num_class); 
hStrings = text(x(:),y(:),textStrings(:), 'HorizontalAlignment','center');
midValue = mean(get(gca,'CLim')); 
textColors = repmat(mat(:) > midValue,1,3); 
set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors
set(gca,'xtick',1:1:16);
set(gca,'xticklabel',tick,'XAxisLocation','top');

rotateXLabels(gca, 315 );% rotate the x tick
set(gca,'ytick',1:1:16);
set(gca,'yticklabel',tick);


