%% 基于超像素分割的高光谱分类
% hyperspectral image classification with small training sample size 
% using superpixel-guided training sample enlargement
% 对每一个超像素体，
% 1）若其中仅仅包含一类训练样本，则将该超像素体均归为该类；
%     这样做不够谨慎，改为将该超像素体内与该训练样本比较相似的归为该类；
% 2）若其中不包含任何已知样本，则利用加权法回归距离分类器，对其进行分类；
%    （当训练样本很少时，会发生这种情况）
% 3）若其中包含多类训练样本，则利用加权法回归距离分类器，
%    将其中各像素划分为这多类中的一类；（当超像素足够多时，这种情况可忽略）
%%  2023-07-10更新
clc
clear
close all
tic;

dir ='..\datasets\';
%% 读数据 %%%%%%%%%%%%%%%%%%%%%%%%%%
%size:Indian(145*145) PaviaU(610*340) SalinasA(512*217) KSC(512*614)
dataNameSet={'Indian_pines','PaviaU','Salinas1120'};
gtNameSet={'Indian_pines_gt','PaviaU_gt','Salinas_gt'};
%SpNums=[200,1600,900,2500]; %10x10,11*11, 10x10 11x11
%SpNums=round([145*145/64 610*340/121 940*475/121]);
SpNums=[300 1600 3690];
LmdSet15=[0.01 0.01 0.0001 0.01];%数据不模1化，距离不归一化，各数据库的参数
%Indian_pines：350=300优于200优于400优于250=250，lambda=0.05，介于[200,400],半径=8
%PaviaU：1700=1600优于1500=1400，lambda=0.01，介于[1500,1600],半径=11.38
%Salinas：918优于1111》1400=1372略优于772,1600，lambda=0.0001, 半径=11，介于[918,1111]
% KSC：2598=2500(1550)略优于2000(1107)》2200(1291)，lambda=0.01 ，介于[2000,2500]，半径=11.2

LmdSets=[1e-3 1e-2 1e-2 1e-2 0.1;
    1e-3 1e-2 1e-2 1e-2 1e-2;
    1e-4 1e-4 1e-4 1e-4 1e-4];

ifNormHSI=0;%是否对光谱数据进行范数归一化处理，影响不太大
ifNormDist=0;%是否对距离数据进行标准化处理，影响较大

expTimes=2;%随机实验次数
Ps=0.1;%indian pines每类训练样本数目
lthP=length(Ps);
classnum=9;   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 类别数
datano = 2;    %%% 数据集选择
OA = zeros(1,expTimes); AA = OA;
KA = OA; accurperclass = zeros(classnum,expTimes);
plabel_STSEDWLR = zeros(610,340,expTimes);     %%%%%%%%%% 图像大小

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
    %% 立方体数据转化为矩阵数据，并归一化%%%%%%%%
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
    %% 进行超像素分割 Superpixel segmentation
    [Sp,nSp]=SuperpixelSegmentation(data,numSuperpixels);
    
    for pth=1:lthP
        P=Ps(pth);%P=[5 10 15 20 30]%每类训练样本数目
        for expT=1:expTimes
            nClass=max(label(:));
            %% 划分训练样本和测试样本
            %         rng(expT*10,'twister');%随机数生成器
            mask=false(row,col);%已知点掩模
            nListTrn=zeros(nClass,1);%第类的训练样本数
            nListClass=zeros(nClass,1);%每类的样本总数
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
                    ntrnc=max(round(P*n),1); %第c类训练样本数
                else
                    ntrnc=P;
                end
                if ntrnc>=n
                    ntrnc=15;
                end
                nListTrn(js)=ntrnc;
                id1=randperm(n,ntrnc);
                mask(id(id1))=true;%已知点掩模，mask(r,c)=true,则(r,c)点为已知点
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
            predictedLabel=zeros(row,col); %预测类别矩阵
            predictedLabel(mask)=labels(mask);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            tic
            %% 首先将超像素内只包含一类训练样本的超像素识别为该类
            SpInfo.unrecg=true(nSp,1);%记录超像素是否已经识别
            SpInfo.gIdx=cell(nSp,1);%存储各超像素的索引
            SpInfo.ntp=zeros(nSp,1,'uint16');%各超像素中包含的训练样本类别数
            SpInfo.types=cell(nSp,1);%各超像素中包含的训练样本类别
            for t=1:nSp
                idt= find(Sp==t);%
                if isempty(idt)%该超像素不用识别
                    SpInfo.unrecg(t)=false;
                    SpInfo.gIdx{t}=[];
                    continue;
                end
                %查看其中是否包含已知类别数
                id1=find(mask(idt));
                ns=numel(id1);
                if ns %其中包含训练样本
                    lablei=labels(idt(id1));
                    types=unique(lablei);
                    ntp=numel(types);
                    if ntp==1 %类别数为1——仅仅包含一类训练样本
                        %将该超像素识别为该训练样本类
                        predictedLabel(idt)=types;
                        SpInfo.unrecg(t)=false;
                        %将已识别超像素作为训练样本
                        mask(idt)=true;
                        continue;
                    end
                    % 记录该超像素信息
                    SpInfo.ntp(t)=ntp;
                    SpInfo.types{t}=types;
                end
                SpInfo.gIdx{t}=idt;
            end
            tm0=toc;
            %% 识别内有多类训练样本或无训练样本的超像素
            idTrn=find(mask);
            [I,J] = ind2sub([row,col],idTrn);
            trnLabel=labels(idTrn);
            A=X(:,idTrn);%训练矩阵
            %构造测试矩阵，由超像素内包含多类训练样本
            % 或不包含训练样本的超像素的均值向量构成
            id=find((SpInfo.ntp>1 | SpInfo.ntp==0)&SpInfo.unrecg);
            nT=numel(id);
            Y=zeros(dim,nT);
            yTypes=cell(nT,1);
            It=zeros(nT,1);
            Jt=zeros(nT,1);
            for t=1:nT
                idt=SpInfo.gIdx{id(t)};%
                Y(:,t)=mean(X(:,idt),2);%第t个超像素体数据集的均值向量
                yTypes{t}=SpInfo.types{id(t)};%第t个超像素体包含的训练样本类别
                [r0,c0]=ind2sub([row,col],idt);%第t个超像素体的坐标
                It(t)=round(mean(r0));%第t个超像素体重心位置行坐标
                Jt(t)=round(mean(c0));%第t个超像素体重心位置纵坐标
            end
            %%
            tstLabel=labels(idTst);
            %ratio和1-ratio为谱距离、空间距离所占比重;
            lambda=LmdSets(nameNb,pth);
            %利用距离加权回归分类器，对其进行分类；
            %%%%%%%%%%%%%%%%直接法%%%%%%%%%%%%%%%%%%%%%%%%%
            predLabel=DWLRC(A,Y,trnLabel,I,J,It,Jt,yTypes,lambda,ifNormDist,0);
            for t=1:nT
                idt=SpInfo.gIdx{id(t)};%
                predictedLabel(idt)=predLabel(t);
            end
            tm1=toc+tm0;
            %% 计算各类识别精度
            [OA(expT), AA(expT), KA(expT), accurperclass(:,expT)]=ClassifyAccuracy(tstLabel,predictedLabel(idTst));
%             [IA2,OA2,AA2]=ComputeAccuracy(predictedLabel(idTst),tstLabel,nClass,nListClass-nListTrn);
            plabel_STSEDWLR(:,:,expT) = predictedLabel;
            
            %         predictedLabel(label==0)=0;%去掉背景
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
%距离加权线性回归分类器
% min||Aix-y||_2^2+lambda||Wx||_2^2;
% Ai'Aix-A'y+lambdaW'Wx=0 =>x=(Ai'Ai+lambdaW'W)\(A'y)
% so Aix-y=Ai*inv(Ai'*Ai+lambda*W'*W)*Ai'y-y
% ratio和1-ratio为谱距离、空间距离所占比重

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
        %%计算加权矩阵
        if  ratio==0
            d=(Ic-r0).^2+(Jc-c0).^2;%空间距离
        else
            d1=sqrt(sum((Ac-repmat(y,1,nck)).^2))';%谱欧式距离
            %d1=1-abs(y'*Ai);%谱角欧式距离
            d2=sqrt((Ic-r0).^2+(Jc-c0).^2);%空间距离
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
        %err=d'*d/(x'*x);%% 等于(||Acx-y||/||x||)^2
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
dist_type = 2; % 1:ED；2：SAD; 3:SID; 4:SAD-SID
seg_all = 1; % 1: All pixels are clustered， 2：exist un-clustered pixels
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
%DataTrains：训练样本数据，[x y p(x,y)]，一行一个样本
%CTrain：the number of training samples of the ith class
% DataTests：测试样本数据，[x y p(x,y)]，一行一个样本
% Spatial-Aware Collaborative Representation for Hyperspectral Remote Sensing Image Classification
% Example code
% Reference
% [1] J. Jiang, C. Chen, Y. Yu, X. Jiang, and J. Ma, “Spatial-Aware CollaborativeRepresentation
%     for Hyperspectral Remote Sensing Image Classification,” IEEE Geoscience and Remote Sensing Letters,
%     vol. 14, no. 3, pp. 404-408, 2017.

DataTrain = DataTrains(:,3:end);%[x y p(x,y)]
DataTest  = DataTests(:,3:end); %[x y p(x,y)]

DDT = DataTrain*DataTrain';

numClass = length(CTrain);
m= size(DataTest,1); %测试样本数
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