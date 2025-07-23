clc,clear

dir = 'datasets\';
addpath(genpath(pwd))
sr = 1;
load([dir,'Indian_pines1100.mat']);
load([dir,'Indian_pines_gt.mat']);
spectral = indian_pines;    %including water absorption bands [104-108], [150-163], 220
[n,m,dim] = size(spectral);
Xlabel = double(reshape(indian_pines_gt,n*m,1));
clear indian_pines indian_pines_gt;

% parpool(4);
tic;
%% data normalization
X = reshape(spectral,n*m,dim);
X = mapminmax(X);
numberoflabel = max(Xlabel);
numberofdata = zeros(1,numberoflabel);
clear spectral
%% claasification results
num = 10;
OA=zeros(1,num); AA=OA;
Ka=OA;accurperclass=zeros(numberoflabel,num);
% rng(2)

X = X(:,1:1:end);

for randi = 1 : num

    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% 选取样本
    ratio = 0.1;   % 训练样本比例
%     tationum = 5;  % 训练样本个数/每类
    traindataNo=[];testdataNo=[];
    for i = 1:numberoflabel
        ind = find(Xlabel==i);
        numberofdata(i) = length(ind);
        if(numberofdata(i) ~= 0)
            No = randperm(numberofdata(i));
            Numper = ceil(numberofdata(i)*ratio)
%             Numper = tationum;
            traindataNo = [traindataNo;ind(No(1:Numper))];
            testdataNo = [testdataNo;ind(No(Numper+1:numberofdata(i)))];
        end
    end
    traindata = X(traindataNo,:);
    testdata = X(testdataNo,:);
    
    trainlabels = Xlabel(traindataNo);
    testlabels= Xlabel(testdataNo);
 
    %% SVM classification with spectral data
    [~, bestc, bestg] = SVMcgForClass(trainlabels,traindata,-10,10,-10,10,10,1,1);
    cmd = ['-c ', num2str(bestc), ' -g ', num2str(bestg), ' -t ', num2str(2)];
%     cmd = ['-c ', num2str(128), ' -g ', num2str(0.0625), ' -t ', num2str(2)];
    model = svmtrain(trainlabels, traindata, cmd);
    type = 2;
    CR = ClassResult(testlabels, testdata, model, type);
    accOSVM(randi) = CR.accuracy(1);  %精度
    label_SVM = Xlabel;
    label_SVM(testdataNo) = CR.plabel;
    plabel_SVM(:,:,randi) = reshape(label_SVM,n,m);
    [OA(randi),AA(randi),Ka(randi),accurperclass(:,randi)] = classficationresult(CR.plabel,testlabels);
    clear model;
 
end

Results(:,sr) = [mean(OA);mean(AA);mean(Ka)];
clear OA AA kappa
toc;
% save indian_SVM.mat OA AA Ka accurperclass plabel_SVM;

% delete(gcp);