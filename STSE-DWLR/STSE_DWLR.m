%% ���ڳ����طָ�ĸ߹��׷���
% hyperspectral image classification with small training sample size 
% using superpixel-guided training sample enlargement
% ��ÿһ���������壬
% 1�������н�������һ��ѵ���������򽫸ó����������Ϊ���ࣻ
%     ������������������Ϊ���ó������������ѵ�������Ƚ����ƵĹ�Ϊ���ࣻ
% 2�������в������κ���֪�����������ü�Ȩ���ع�����������������з��ࣻ
%    ����ѵ����������ʱ���ᷢ�����������
% 3�������а�������ѵ�������������ü�Ȩ���ع�����������
%    �����и����ػ���Ϊ������е�һ�ࣻ�����������㹻��ʱ����������ɺ��ԣ�
%%  2023-07-10����
clc
clear
close all
tic;

dir ='..\datasets\';
%% ������ %%%%%%%%%%%%%%%%%%%%%%%%%%
%size:Indian(145*145) PaviaU(610*340) SalinasA(512*217) KSC(512*614)
dataNameSet={'Indian_pines','PaviaU','Salinas1120'};
gtNameSet={'Indian_pines_gt','PaviaU_gt','Salinas_gt'};
%SpNums=[200,1600,900,2500]; %10x10,11*11, 10x10 11x11
%SpNums=round([145*145/64 610*340/121 940*475/121]);
SpNums=[300 1600 3690];
LmdSet15=[0.01 0.01 0.0001 0.01];%���ݲ�ģ1�������벻��һ���������ݿ�Ĳ���
%Indian_pines��350=300����200����400����250=250��lambda=0.05������[200,400],�뾶=8
%PaviaU��1700=1600����1500=1400��lambda=0.01������[1500,1600],�뾶=11.38
%Salinas��918����1111��1400=1372������772,1600��lambda=0.0001, �뾶=11������[918,1111]
% KSC��2598=2500(1550)������2000(1107)��2200(1291)��lambda=0.01 ������[2000,2500]���뾶=11.2

LmdSets=[1e-3 1e-2 1e-2 1e-2 0.1;
    1e-3 1e-2 1e-2 1e-2 1e-2;
    1e-4 1e-4 1e-4 1e-4 1e-4];

ifNormHSI=0;%�Ƿ�Թ������ݽ��з�����һ������Ӱ�첻̫��
ifNormDist=0;%�Ƿ�Ծ������ݽ��б�׼������Ӱ��ϴ�

expTimes=2;%���ʵ�����
Ps=0.1;%indian pinesÿ��ѵ��������Ŀ
lthP=length(Ps);
classnum=9;   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% �����
datano = 2;    %%% ���ݼ�ѡ��
OA = zeros(1,expTimes); AA = OA;
KA = OA; accurperclass = zeros(classnum,expTimes);
plabel_STSEDWLR = zeros(610,340,expTimes);     %%%%%%%%%% ͼ���С

nameNb=datano;
numSuperpixels=SpNums(nameNb);
%%
dataName=dataNameSet{nameNb};
gtName=gtNameSet{nameNb};
Data = cell2mat(struct2cell(load([dir,dataName])));
label = cell2mat(struct2cell(load([dir,gtName])));

Results = zeros(3,10);

for sstep = 10 : 10
    data = Data(:,:,1:sstep:end);
    [row,col,dim]=size(data);
    nPixel=row*col;
    %% ����������ת��Ϊ�������ݣ�����һ��%%%%%%%%
    X=zeros(dim,nPixel);
    js=1;
    for c=1:col
        for r=1:row
            x=reshape(data(r,c,:),dim,1);
            m=min(x);
            tmp=(x-m)/(max(x)-m);
            if ifNormHSI
                X(:,js)=tmp/norm(tmp);
            else
                X(:,js)=tmp;
            end
            js=js+1;
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% ���г����طָ� Superpixel segmentation
    [Sp,nSp]=SuperpixelSegmentation(data,numSuperpixels);
    
    for pth=1:lthP
        P=Ps(pth);%P=[5 10 15 20 30]%ÿ��ѵ��������Ŀ
        for expT=1:expTimes
            nClass=max(label(:));
            %% ����ѵ�������Ͳ�������
            %         rng(expT*10,'twister');%�����������
            mask=false(row,col);%��֪����ģ
            nListTrn=zeros(nClass,1);%�����ѵ��������
            nListClass=zeros(nClass,1);%ÿ�����������
            idTst=[];
            labels=label;
            js=1;
            for c=1:nClass
                id=find(label==c);
                n=numel(id);
                if ~n,continue;end
                nListClass(js)=n;
                labels(id)=js;
                if P<1
                    ntrnc=max(round(P*n),1); %��c��ѵ��������
                else
                    ntrnc=P;
                end
                if ntrnc>=n
                    ntrnc=15;
                end
                nListTrn(js)=ntrnc;
                id1=randperm(n,ntrnc);
                mask(id(id1))=true;%��֪����ģ��mask(r,c)=true,��(r,c)��Ϊ��֪��
                id(id1)=[];
                idTst=[idTst; id];
                js=js+1;
            end
            %%%%
            nClass=js-1;
            nListTrn(js:end)=[];
            nListClass(js:end)=[];
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %idTrn0=find(mask);
            predictedLabel=zeros(row,col); %Ԥ��������
            predictedLabel(mask)=labels(mask);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            tic
            %% ���Ƚ���������ֻ����һ��ѵ�������ĳ�����ʶ��Ϊ����
            SpInfo.unrecg=true(nSp,1);%��¼�������Ƿ��Ѿ�ʶ��
            SpInfo.gIdx=cell(nSp,1);%�洢�������ص�����
            SpInfo.ntp=zeros(nSp,1,'uint16');%���������а�����ѵ�����������
            SpInfo.types=cell(nSp,1);%���������а�����ѵ���������
            for t=1:nSp
                idt= find(Sp==t);%
                if isempty(idt)%�ó����ز���ʶ��
                    SpInfo.unrecg(t)=false;
                    SpInfo.gIdx{t}=[];
                    continue;
                end
                %�鿴�����Ƿ������֪�����
                id1=find(mask(idt));
                ns=numel(id1);
                if ns %���а���ѵ������
                    lablei=labels(idt(id1));
                    types=unique(lablei);
                    ntp=numel(types);
                    if ntp==1 %�����Ϊ1������������һ��ѵ������
                        %���ó�����ʶ��Ϊ��ѵ��������
                        predictedLabel(idt)=types;
                        SpInfo.unrecg(t)=false;
                        %����ʶ��������Ϊѵ������
                        mask(idt)=true;
                        continue;
                    end
                    % ��¼�ó�������Ϣ
                    SpInfo.ntp(t)=ntp;
                    SpInfo.types{t}=types;
                end
                SpInfo.gIdx{t}=idt;
            end
            tm0=toc;
            %% ʶ�����ж���ѵ����������ѵ�������ĳ�����
            idTrn=find(mask);
            [I,J] = ind2sub([row,col],idTrn);
            trnLabel=labels(idTrn);
            A=X(:,idTrn);%ѵ������
            %������Ծ����ɳ������ڰ�������ѵ������
            % �򲻰���ѵ�������ĳ����صľ�ֵ��������
            id=find((SpInfo.ntp>1 | SpInfo.ntp==0)&SpInfo.unrecg);
            nT=numel(id);
            Y=zeros(dim,nT);
            yTypes=cell(nT,1);
            It=zeros(nT,1);
            Jt=zeros(nT,1);
            for t=1:nT
                idt=SpInfo.gIdx{id(t)};%
                Y(:,t)=mean(X(:,idt),2);%��t�������������ݼ��ľ�ֵ����
                yTypes{t}=SpInfo.types{id(t)};%��t���������������ѵ���������
                [r0,c0]=ind2sub([row,col],idt);%��t���������������
                It(t)=round(mean(r0));%��t��������������λ��������
                Jt(t)=round(mean(c0));%��t��������������λ��������
            end
            %%
            tstLabel=labels(idTst);
            %ratio��1-ratioΪ�׾��롢�ռ������ռ����;
            lambda=LmdSets(nameNb,pth);
            %���þ����Ȩ�ع��������������з��ࣻ
            %%%%%%%%%%%%%%%%ֱ�ӷ�%%%%%%%%%%%%%%%%%%%%%%%%%
            predLabel=DWLRC(A,Y,trnLabel,I,J,It,Jt,yTypes,lambda,ifNormDist,0);
            for t=1:nT
                idt=SpInfo.gIdx{id(t)};%
                predictedLabel(idt)=predLabel(t);
            end
            tm1=toc+tm0;
            %% �������ʶ�𾫶�
            [OA(expT), AA(expT), KA(expT), accurperclass(:,expT)]=ClassifyAccuracy(tstLabel,predictedLabel(idTst));
%             [IA2,OA2,AA2]=ComputeAccuracy(predictedLabel(idTst),tstLabel,nClass,nListClass-nListTrn);
            plabel_STSEDWLR(:,:,expT) = predictedLabel;
            
            %         predictedLabel(label==0)=0;%ȥ������
            %         tmp=label2rgb(predictedLabel,'jet','k');
            %         tmp=label2rgb(predictedLabel,'hsv','k');
            %         figure,imshow(tmp,[])
        end%end of for expT
    end%end of for P
    Results(:,11-sstep) = [mean(OA);mean(AA);mean(KA)];
end%end of for nameNb
toc;
% save honghu_STSEDWLR.mat OA AA K accurperclass plabel_STSEDWLR;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [OA, AA, KA, IA]=ClassifyAccuracy(true_label,estim_label)
% function ClassifyAccuracy(true_label,estim_label)
% This function compute the confusion matrix and extract the OA, AA
% and the Kappa coefficient.
%http://kappa.chez-alice.fr/kappa_intro.htm

l=length(true_label);
nb_c=max(true_label);

%compute the confusion matrix
confu=zeros(nb_c);
for i=1:l
    confu(true_label(i),estim_label(i))= confu(true_label(i),estim_label(i))+1;
end

OA=trace(confu)/sum(confu(:)); %overall accuracy
IA=diag(confu)./sum(confu,2);  %class accuracy
IA(isnan(IA))=0;
number=size(IA,1);

AA=sum(IA)/number;
Po=OA;
Pe=(sum(confu)*sum(confu,2))/(sum(confu(:))^2);
KA=(Po-Pe)/(1-Pe);%kappa coefficient
OA = OA*100;
AA = AA*100;
IA = IA*100;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [IA, OA, AA]=ComputeAccuracy(predictedLabel,tstLabel,nClass,nListTst)
CM = confusionmat(tstLabel,predictedLabel);
IA=zeros(1,nClass);
bool=predictedLabel~=tstLabel;
for c=1:nClass
    id=find(bool & tstLabel==c);
    IA(c)=100-numel(id)/nListTst(c)*100;
end
OA=sum(IA'.*nListTst)/sum(nListTst);
AA=mean(IA);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function predictedLabel=DWLRC(A,Y,trnLabel,I,J,It,Jt,yTypes,lambda,ifNorm,ratio)
%�����Ȩ���Իع������
% min||Aix-y||_2^2+lambda||Wx||_2^2;
% Ai'Aix-A'y+lambdaW'Wx=0 =>x=(Ai'Ai+lambdaW'W)\(A'y)
% so Aix-y=Ai*inv(Ai'*Ai+lambda*W'*W)*Ai'y-y
% ratio��1-ratioΪ�׾��롢�ռ������ռ����

if nargin==9
    ratio=0;
    ifNorm=false;
elseif nargin==10
    ratio=0;
end
nClass=max(trnLabel);
nTst=numel(It);
predictedLabel=zeros(nTst,1);
for t=1:nTst
    r0=It(t);
    c0=Jt(t);
    y=Y(:,t);
    err0=inf;
    if isempty(yTypes{t})
        classArray=1:nClass;
        nc=nClass;
    else
        classArray=yTypes{t};
        nc=numel(classArray);
    end
    for k=1:nc
        c=classArray(k);
        id=trnLabel==c;
        Ac=A(:,id);
        Ic=I(id);
        Jc=J(id);
        nck=numel(Ic);
        %%�����Ȩ����
        if  ratio==0
            d=(Ic-r0).^2+(Jc-c0).^2;%�ռ����
        else
            d1=sqrt(sum((Ac-repmat(y,1,nck)).^2))';%��ŷʽ����
            %d1=1-abs(y'*Ai);%�׽�ŷʽ����
            d2=sqrt((Ic-r0).^2+(Jc-c0).^2);%�ռ����
            d=(ratio*d1+(1-ratio)*d2).^2;
        end
        if ifNorm
            %d=d/max(d);
            d=nck*d/sum(d);
        end
        W=diag(lambda*d);
        %%
        x=(Ac'*Ac+W)\(Ac'*y);
        d=Ac*x-y;
        %err=d'*d/(x'*x);%% ����(||Acx-y||/||x||)^2
        err=d'*d;
        if err<err0
            err0=err;
            predictedLabel(t)=c;
        end
    end
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [labels_t,numlabels]=SuperpixelSegmentation(data,numSuperpixels)

[nl, ns, nb] = size(data);
x = data;
x = reshape(x, nl*ns, nb);
x = x';

input_img = zeros(1, nl * ns * nb);
startpos = 1;
for i = 1 : nl
    for j = 1 : ns
        input_img(startpos : startpos + nb - 1) = data(i, j, :);
        startpos = startpos + nb;
    end
end


%% perform Regional Clustering

%numSuperpixels = 200;  % number of segments
compactness = 0.1; % compactness2 = 1-compactness, compactness*dxy+compactness2*dspectral
dist_type = 2; % 1:ED��2��SAD; 3:SID; 4:SAD-SID
seg_all = 1; % 1: All pixels are clustered�� 2��exist un-clustered pixels
% labels:segment no of each pixel
% numlabels: actual number of segments
[labels, numlabels, ~, ~] = RCSPP(input_img, nl, ns, nb, numSuperpixels, compactness, dist_type, seg_all);
clear input_img;

labels_t = zeros(nl, ns, 'int32');
for i=1:nl
    for j=1:ns
        labels_t(i,j) = labels((i-1)*ns+j);
    end
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function DisplaySuperpixelSegmentation(Sp,imfile,row,col)
img=imread(imfile);

tmp=Sp';
inlabels=int32(tmp(:))';

[segment_out] = DrawContoursAroundSegments(inlabels, row, col);
tmp=zeros(col,row,'int32');
tmp(:)=segment_out;
tmp=tmp';

img(tmp==1)=255;
img(tmp==0)=0;
imshow(img,[])
imwrite(img,'DisplaySuperpixel.tiff','tiff')
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function class=JSaCR(DataTrains, CTrain, DataTests, lambda, c, gamma)
%DataTrains��ѵ���������ݣ�[x y p(x,y)]��һ��һ������
%CTrain��the number of training samples of the ith class
% DataTests�������������ݣ�[x y p(x,y)]��һ��һ������
% Spatial-Aware Collaborative Representation for Hyperspectral Remote Sensing Image Classification
% Example code
% Reference
% [1] J. Jiang, C. Chen, Y. Yu, X. Jiang, and J. Ma, ��Spatial-Aware CollaborativeRepresentation
%     for Hyperspectral Remote Sensing Image Classification,�� IEEE Geoscience and Remote Sensing Letters,
%     vol. 14, no. 3, pp. 404-408, 2017.

DataTrain = DataTrains(:,3:end);%[x y p(x,y)]
DataTest  = DataTests(:,3:end); %[x y p(x,y)]

DDT = DataTrain*DataTrain';

numClass = length(CTrain);
m= size(DataTest,1); %����������
for j = 1: m
    %     if mod(j,round(m/20))==0
    %         fprintf('*...');
    %     end
    
    xy = DataTests(j, 1:2);
    XY = DataTrains(:, 1:2);
    norms = sum((abs(XY' - repmat(xy', [1 size(XY,1)]))).^c);
    norms = norms./max(norms);
    D = diag(gamma.*norms);
    
    Y = DataTest(j, :); % 1 x dim
    norms = sum((DataTrain' - repmat(Y', [1 size(DataTrain,1)])).^2);
    % norms = ones(size(DataTrain,1), 1);
    G = diag(lambda.*norms);
    weights = (DDT +  G + D)\(DataTrain*Y');
    
    a = 0;
    for i = 1: numClass
        % Obtain Multihypothesis from training data
        HX = DataTrain((a+1): (CTrain(i)+a), :); % sam x dim
        HW = weights((a+1): (CTrain(i)+a));
        a = CTrain(i) + a;
        Y_hat = HW'*HX;
        
        Dist_Y(j, i) = norm(Y - Y_hat);
    end
    Dist_Y(j, :) = Dist_Y(j, :)./sum(Dist_Y(j, :));
end
[~, class] = min(Dist_Y');
end