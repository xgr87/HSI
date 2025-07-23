function [InImg, V] = DLNet_train(InImg,TrnLabels,Tr_idx_R,Tr_idx_C,DLNet,Te_idx_R,Te_idx_C,TestLabels)

addpath('./Utils')
filsize=7;
InImg2=[];
In1=InImg;
if length(DLNet.NumFilters)~= DLNet.NumStages;
    display('Length(DLNet.NumFilters)~=DLNet.NumStages')
    return
end

NumImg = numel(Tr_idx_R);

V = cell(DLNet.NumStages,1);

NumStage=DLNet.NumStages;

for i=1:NumStage
    if(rem(i,2))
        display(['Computing filter bank and its outputs at stage ' num2str(i) '...'])
        % pa1=[1000 1000 1000 1000 1000 1000];
        % % pa1=[600 300 300 200]; %0.4
        % pa2=[0.1 0.1 0.1 0.1 0.1 0.1];
        
        
        [r1 c1 d1]=size(InImg);
        ftest=[];
        ftrain=[];
        
        for i1=1:numel(Tr_idx_R)
            ftrain(:,i1)=reshape(InImg(Tr_idx_R(i1),Tr_idx_C(i1),:),size(InImg,3),1);
        end
        
        
        for i1=1:r1
            for j1=1:c1
                ftest=[ftest reshape(InImg(i1,j1,:),size(InImg,3),1)];
            end
        end
        
        TestLabels=ones(1,r1*c1);
        % T = elm_kernel_new(ftrain,TrnLabels,ftest,TestLabels1,1, pa1(fix(i/2)+1), 'RBF_kernel',pa2(fix(i/2)+1));
        C = 2^-30; s = 0.5;%the l2 regularization parameter and the shrinkage scale of the enhancement nodes
        N11=15;%feature nodes  per window
        N2=20;% number of windows of feature nodes
        N33=6000;% number of enhancement nodes
        N1=N11; N3=N33;
        
        temp_Tr=zeros(numel(unique(TrnLabels)), numel(TrnLabels));
        for iii = 1:numel(TrnLabels)
            temp_Tr(TrnLabels(iii),iii)=1;
        end
        Tr=temp_Tr*2-1;
        
        temp_Te=zeros(numel(unique(TrnLabels)), numel(TestLabels));
        for iii = 1:numel( TestLabels)
            temp_Te(TestLabels(iii),iii)=1;
        end
        Te=temp_Te*2-1;
        T= bls_train_new(ftrain',Tr',ftest',Te',s,C,N1,N2,N3);
        % T=T';
        % CC=(T-ones(numel(unique(TrnLabels)),1)*min(T));
        % CC=CC./(ones(numel(unique(TrnLabels)),1)*sum(CC));
        % T=CC;
        tt=1;
        Txx=(T-min(T,[],2)*ones(1,size(T,2)));
        
        T=Txx./(sum(Txx,2)*ones(1,size(T,2)));
        
        InImg=zeros(r1,c1,size(T,2));
        for i1=1:r1
            for j1=1:c1
                InImg(i1,j1,:)=T(tt,:);
                tt=tt+1;
            end
        end
    else
        display(['Computing PCA filter bank and its outputs at stage ' num2str(i) '...'])
        
        % [fSiz,filters,c1OL,numSimpleFilters] = init_gabor_New([0 22.5 45 67.5 90 112.5 135 157.5], filsize);
        [fSiz,filters,c1OL,numSimpleFilters] = init_gabor_New([0 22.5 45 67.5 90 112.5 135 157.5], filsize);
        Fea_Gabor=zeros(r1,c1,128);
        num=1;
        
        InImg1=InImg;
        %   for ii=1:r1
        %       for j=1:c1
        %           temp=reshape(InImg1(ii,j,:),1,numel(unique(TrnLabels)));
        %           InImg1(ii,j,:)=temp/norm(temp);
        %       end
        %   end
        
        for ii=1:8
            for j=1:16
                Fea_Gabor(:,:,num) = filter2(reshape(filters(:,ii),filsize,filsize),InImg1(:,:,j),'same');
                num=num+1;
            end
        end
        for ii=1:r1
            for j=1:c1
                temp=reshape(Fea_Gabor(ii,j,:),1,128);
                Fea_Gabor(ii,j,:)=temp/norm(temp);
            end
        end
        para=0.7;
        InImg=cat(3,para*In1,(1-para)*Fea_Gabor);
        %  InImg=cat(3,In1,Fea_Gabor);
    end
end









