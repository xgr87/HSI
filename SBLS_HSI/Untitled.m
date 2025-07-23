t1=double(cdata1);
t=double(cdata);
for i=1:610
for j=1:340
if (t1(i,j)==t(i,j))
    
elseif(t(i,j)~=0)   
   t1(i,j)=t(i,j);
end
end
end

numel(find(t1>0))