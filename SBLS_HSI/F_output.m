function OutImg= F_output(InImg,NumFilters)

[Im1 Im2 Im3]=size(InImg);

OutImg=zeros(Im1, Im2, Im3*NumFilters);

fil=[3 5 7 9 11 13 15 17 19 21 23];

 

for i=1:NumFilters
    S=fil(i);
    temp=zeros(Im1,Im2,Im3);
    InImg1=padarray(InImg,[floor(S/2) floor(S/2)],'symmetric','both' );
    
    for j=1:size(InImg1,1)-S+1
       
        for k=1:size(InImg1,2)-S+1
            
            temp0=InImg1(j:j+S-1,k:k+S-1,:);
            temp1=[];
            temp2=reshape(temp0(round(S/2),round(S/2),:),Im3,1);
            sc=zeros(1,S^2);
            t=1;
            for m=1:S
               for n=1:S
                temp1=[temp1 reshape(temp0(m,n,:),Im3,1)];
                sc(t)=sum((temp1(:,end)-temp2).^2);
                t=t+1;
               end
            end
       
            
            B=std(sc);
%               siga=(B(2))+0.000001;
siga=1/B;
%             siga=0.4;
            W0=exp(-(sc/(siga)));
            [cc dd]=sort(W0);
            W0(dd(1:ceil(S*S*0.7)))=0;
            W0=W0/sum(W0);
%             [cc dd]=sort(W0);
%             W(1:S*(-1))=0;
            xxx=temp1*W0';
            temp(j,k,:)=xxx;
        end
        
    end
    
    OutImg(:,:,(i-1)*Im3+1:i*Im3)=temp;
end

