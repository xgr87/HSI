clc,clear

dir = '..\datasets\';
addpath(genpath(pwd))

load([dir,'Indian_pines440.mat']);
load([dir,'Indian_pines_gt.mat']);

parpool(4);
Result_FSVM = zeros(3,10);
Result_PCA = Result_FSVM;
Result_FPCA = Result_FSVM;
Result_LDA = Result_FSVM;
Result_FDA = Result_FSVM;

for sstep = 1:1
    
    spectral = indian_pines(:,:,1:sstep:end);
    [n,m,dim] = size(spectral);
    Xlabel = double(reshape(indian_pines_gt,n*m,1));
    
    % load([dir,'Salinas_corrected.mat']);
    % load([dir,'Salinas_gt.mat']);
    % spectral = salinas_corrected;
    % [n,m,dim] = size(spectral);
    % Xlabel = double(reshape(salinas_gt,n*m,1));
    
    % load([dir,'PaviaU.mat']);
    % load([dir,'PaviaU_gt.mat']);
    % spectral = paviaU;
    % [n,m,dim] = size(spectral);
    % Xlabel = double(reshape(paviaU_gt,n*m,1));
    %% spatial information extraction
%     r = 7;
%     extendspecral3D = zeros(n+2*r,m+2*r,dim);
%     extendspecral3D(r+1:n+r,r+1:m+r,:) = spectral;
%     extendspecral3D(1:r,r+1:m+r,:) = spectral(r:-1:1,1:m,:);
%     extendspecral3D(n+r+1:n+2*r,r+1:m+r,:) = spectral(n:-1:n-r+1,1:m,:);
%     extendspecral3D(:,1:r,:) = extendspecral3D(:,2*r:-1:r+1,:);
%     extendspecral3D(:,m+r+1:m+2*r,:) = extendspecral3D(:,m+1:m+r,:);
%     
%     spatial3D = zeros(n,m,dim);
%     for i = 1 : n
%         for j = 1 : m
%             spatial3D(i,j,:) = sum(sum(extendspecral3D(i:i+2*r,j:j+2*r,:),1),2)/(2*r+1)^2;
%         end
%     end
    
    %% data normalization
    img = double(spectral);
    X = reshape(spectral,n*m,dim);
    X = mapminmax(X);
    
%     Z = reshape(spatial3D,n*m,dim);
%     Z  = mapminmax(Z);
    
    %% claasification results
    OA=struct(); AA=struct();
    K=struct();accurperclass=struct();
    numberoflabel = max(Xlabel);
    % tic;
    for randi = 1 : 10
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% 选取样本
        ratio = 0.1;   % 训练样本比例
        % tationum = 10;  % 训练样本个数/每类
        traindataNo=[];testdataNo=[];
        numberofdata = zeros(1,numberoflabel);
        Labelmatrix = zeros(n,m);
        for i = 1:numberoflabel
            ind = find(Xlabel==i);
            numberofdata(i) = length(ind);
            if(numberofdata(i) ~= 0)
                No = randperm(numberofdata(i));
                Numper = ceil(numberofdata(i)*ratio);
                traindataNo = [traindataNo;ind(No(1:Numper))];
                testdataNo = [testdataNo;ind(No(Numper+1:numberofdata(i)))];
            end
        end
        traindata = X(traindataNo,:);
        testdata = X(testdataNo,:);
        
%         traindataZ = Z(traindataNo,:);
%         testdataZ = Z(testdataNo,:);
        
        trainlabels = Xlabel(traindataNo);
        testlabels= Xlabel(testdataNo);
        %     for  i = 1 :  length(traindataNo)
        %         XX(i,randi) = find(GroundT(:,1) == traindataNo(i));
        %     end
        
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [numtrain,~] = size(traindata);   %训练样本大小和维数
        [numtest,~] = size(testdata);      %测试样本大小和维数
        sample = [traindata; testdata]';
%         sampleZ = [traindataZ; testdataZ]';
%         sampleT = [sample;sampleZ];
        
%         allsample = [traindata; testdata]'; %加入背景
%         allsampleZ = [traindataZ; testdataZ]';   %加入背景
%         allsampleT = [allsample;allsampleZ];
        num = numtrain + numtest;
%         dim1 = size(sampleT,1);
        %% functional data fit
        % 生成B样条基函数
        r = 10^(-3);
        rangewavelength = r * [1, dim]';  %波长取值范围
%         rangewavelength1 = r * [1, dim1]';  %波长取值范围
        norder = 4;                % B样条阶数
        % 连续分布的数据
        wavelength = r * (1:dim)';   % 波长
        nbasis = norder + length(wavelength) - 2;
%         wavelength1 = r * (1:dim1)';   % 波长
%         nbasis1 = norder + length(wavelength1) - 2;
        Lfd = 2;   %Lfd为粗糙惩罚项的导数的阶数
        
        basisobj = create_bspline_basis(rangewavelength, nbasis, norder, wavelength);
%         basisobj1 = create_bspline_basis(rangewavelength1, nbasis1, norder, wavelength1);
        
        %  名称：定义域，物质，辐射值
        fdnames{1} = 'Wavelength';
        fdnames{2} = 'Substance';
        fdnames{3} = 'Radiance';
        
        % 选取lambda
        lnlam = -8:1:0;    %lambda的取值范围
        gcvsave = zeros(length(lnlam),1);
        parfor i=1:length(lnlam)
            fdParobj = fdPar(basisobj, Lfd, 10^lnlam(i));
%             fdParobj1 = fdPar(basisobj1, Lfd, 10^lnlam(i));
            % 计算拟合误差gcv
            [~, ~, gcv] = smooth_basis(wavelength, sample, fdParobj, [], fdnames);   %the default is 'method' =  'chol', but if 'method' = 'qr', the qr decomposition is used.
%             [~, ~, allgcv] = smooth_basis(wavelength, allsample, fdParobj, [], fdnames);
%             [~, ~, gcvZ] = smooth_basis(wavelength, sampleZ, fdParobj, [], fdnames);
%             [~, ~, allgcvZ] = smooth_basis(wavelength, allsampleZ, fdParobj, [], fdnames);
%             [~, ~, gcvT] = smooth_basis(wavelength1, sampleT, fdParobj1, [], fdnames);
%             [~, ~, allgcvT] = smooth_basis(wavelength1, allsampleT, fdParobj1, [], fdnames);
            gcvsave(i) = sum(gcv);
%             allgcvsave(i) = sum(allgcv);
%             gcvsaveZ(i) = sum(gcvZ);
%             allgcvsaveZ(i) = sum(allgcvZ);
%             gcvsaveT(i) = sum(gcvT);
%             allgcvsaveT(i) = sum(allgcvT);
        end
        [~, k1] = max(-gcvsave);   %计算所有样本拟合误差最小的lambda。
%         [~, k2] = max(-allgcvsave);
%         [~, k3] = max(-gcvsaveZ);
%         [~, k4] = max(-allgcvsaveZ);
%         [~, k5] = max(-gcvsaveT);
%         [~, k6] = max(-allgcvsaveT);
        lambda = 10^lnlam(k1);  % 最优lambda
%         alllambda = 10^lnlam(k2);
%         lambdaZ = 10^lnlam(k3);
%         alllambdaZ = 10^lnlam(k4);
%         lambdaT = 10^lnlam(k5);
%         alllambdaT = 10^lnlam(k6);
        %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % 粗糙惩罚法
        fdParobj = fdPar(basisobj, Lfd, lambda);
%         allfdParobj = fdPar(basisobj, Lfd, alllambda);
%         fdParobjZ = fdPar(basisobj, Lfd, lambdaZ);
%         allfdParobjZ = fdPar(basisobj, Lfd, alllambdaZ);
%         fdParobjT = fdPar(basisobj1, Lfd, lambdaT);
%         allfdParobjT = fdPar(basisobj1, Lfd, alllambdaT);
        
        [fdobj, df, gcv] = smooth_basis(wavelength, sample, fdParobj, [], fdnames);
%         [allfdobj, alldf, allgcv] = smooth_basis(wavelength, allsample, allfdParobj, [], fdnames);
%         [fdobjZ, dfZ, gcvZ] = smooth_basis(wavelength, sampleZ, fdParobjZ, [], fdnames);
%         [allfdobjZ, alldfZ, allgcvZ] = smooth_basis(wavelength, allsampleZ, allfdParobjZ, [], fdnames);
%         [fdobjT, dfT, gcvT] = smooth_basis(wavelength1, sampleT, fdParobjT, [], fdnames);
%         [allfdobjT, alldfT, allgcvT] = smooth_basis(wavelength1, allsampleT, allfdParobjT, [], fdnames);
        
        fdmat = eval_fd(wavelength, fdobj);    %获取拟合之后的数据
        coef   = getcoef(fdobj);  %取基函数的系数
        
%         allfdmat = eval_fd(wavelength, allfdobj);    %获取拟合之后的数据
%         allcoef   = getcoef(allfdobj);  %取基函数的系数
%         
%         fdmatZ = eval_fd(wavelength, fdobjZ);
%         coefZ   = getcoef(fdobjZ);
%         
%         allfdmatZ = eval_fd(wavelength, allfdobjZ);
%         allcoefZ   = getcoef(allfdobjZ);
%         
%         fdmatT = eval_fd(wavelength1, fdobjT);
%         coefT   = getcoef(fdobjT);
%         
%         allfdmatT = eval_fd(wavelength1, allfdobjT);
%         allcoefT   = getcoef(allfdobjT);
       %% SVM classification with spectral-spatial original data
%         %     Odatatrain = sampleT(:,1:numtrain)';
%         %     Odatatest  = sampleT(:,1+numtrain : num)';
%         
%         %     Odatatrain = sample(:,1:numtrain)';          % only uses the spectral information
%         %     Odatatest  = sample(:,1+numtrain : num)';
%         %
%         %     [~, bestc, bestg] = SVMcgForClass(trainlabels,Odatatrain,-10,10,-10,10,10,1,1);
%         %     cmd0 = ['-c ', num2str(bestc), ' -g ', num2str(bestg), ' -t ', num2str(2)];
%         % %     cmd0 = ['-c ', num2str(1000), ' -g ', num2str(1), ' -t ', num2str(2)];   % only uses the spectral information
%         %     model = svmtrain(trainlabels, Odatatrain, cmd0);
%         %     type = 2;
%         %     CR0 = ClassResult(testlabels, Odatatest, model, type);  %测试样本分类结果
%         %     bgCR0 = ClassResult(allbglabel, allsampleT(:,allbgNo)', model, type);  %背景样本分类结果
%         %
%         %     accOSVM(randi) = CR0.accuracy(1);  %精度
%         %     label_OSVM = Xlabel;
%         %     label_OSVM(testdataNo) = CR0.plabel;
%         %     label_OSVM(allbgNo) = bgCR0.plabel;
%         %     plabel_OSVM(:,:,randi) = reshape(label_OSVM,n,m);
%         %     [OA.OSVM(randi),AA.OSVM(randi),K.OSVM(randi),accurperclass.OSVM(:,randi)] = classficationresult(CR0.plabel,testlabels);
%         %     clear model;
%         %      %% SVM classification after denoising
%         %     Ddatatrain = fdmatT(:,1:numtrain)';
%         %     Ddatatest  = fdmatT(:,1+numtrain : num)';
%         %
%         %     %     [~, bestc, bestg] = SVMcgForClass(trainlabels,fdatatrain,-10,10,-10,10,10,1,1);
%         %     %     cmd1 = ['-c ', num2str(bestc), ' -g ', num2str(bestg), ' -t ', num2str(2)];
%         %     cmd01 = ['-c ', num2str(100), ' -g ', num2str(0.1), ' -t ', num2str(2)];
%         %     model = svmtrain(trainlabels, Ddatatrain, cmd01);
%         %     type = 2;
%         %     CR01 = ClassResult(testlabels, Ddatatest, model, type);
%         %     accDSVM(randi) = CR01.accuracy(1);  %精度
%         %     label_DSVM = Xlabel;
%         %     label_DSVM(testdataNo) = CR01.plabel;
%         %     plabel_DSVM(:,:,randi) = reshape(label_DSVM,n,m);
%         %     [OA.DSVM(randi),AA.DSVM(randi),K.DSVM(randi),accurperclass.DSVM(:,randi)] = classficationresult(CR01.plabel,testlabels);
        %% FSVM classification with spectral-spatial functional data
        %     fdatatrain = coefT(:,1:numtrain)';
        %     fdatatest  = coefT(:,1+numtrain : num)';
        
        fdatatrain = coef(:,1:numtrain)';      % only uses the spectral information
        fdatatest  = coef(:,1+numtrain : num)';
        [~, bestc, bestg] = SVMcgForClass(trainlabels,fdatatrain,-10,10,-10,10,10,1,1);
        cmd1 = ['-c ', num2str(bestc), ' -g ', num2str(bestg), ' -t ', num2str(2)];
        %     cmd1 = ['-c ', num2str(100), ' -g ', num2str(1), ' -t ', num2str(2)];  % only uses the spectral information
        model = svmtrain(trainlabels, fdatatrain, cmd1);
        type = 2;
        CR1 = ClassResult(testlabels, fdatatest, model, type);
        %     bgCR1 = ClassResult(allbglabel, allcoefT(:,allbgNo)', model, type);  %背景样本分类结果
        %     accFSVM(randi) = CR1.accuracy(1);  %精度
        %     label_FSVM = Xlabel;
        %     label_FSVM(testdataNo) = CR1.plabel;
        %     label_FSVM(allbgNo) = bgCR1.plabel;
        %     plabel_FSVM(:,:,randi) = reshape(label_FSVM,n,m);
        [OA.FSVM(randi),AA.FSVM(randi),K.FSVM(randi),accurperclass.FSVM(:,randi)] = classficationresult(CR1.plabel,testlabels);
%         %% PCA+SVM
%         %     [COEFF,SCORE] = pca(sampleT');
%         %     [COEFF,SCORE] = pca(allsampleT');
%         
%         [COEFF,SCORE,latent] = pca(allsample');   % only uses the spectral information
%         latent1=100*latent/sum(latent);
%         num_PC = find(cumsum(latent1) >= 99, 1);   % 貢獻率99%
%         PC = SCORE(:,1:num_PC);
%         train_final = PC(1:numtrain, :);  %训练样本的函数型特征
%         test_final = PC(numtrain + 1:num, :);   %测试样本的函数型特征
%         %     bg_final = PC(num + 1:end, :);   %背景样本的函数型特征
%         %svm分类
%         [~, bestc, bestg] = SVMcgForClass(trainlabels,train_final,-10,10,-10,10,5,1,1);
%         cmd2 = ['-c ', num2str(bestc), ' -g ', num2str(bestg), ' -t ', num2str(2)];
%         %     cmd2 = ['-c ', num2str(100), ' -g ', num2str(1), ' -t ', num2str(2)];
%         model = svmtrain(trainlabels, train_final, cmd2);
%         type = 2;
%         CR2 = ClassResult(testlabels, test_final, model, type);
%         %     bgCR2 = ClassResult(allbglabel, bg_final, model, type);
%         %     accPCA(randi) = CR2.accuracy(1);  %精度
%         %     label_PCA = Xlabel;
%         %     label_PCA(testdataNo) = CR2.plabel;
%         %     label_PCA(allbgNo) = bgCR2.plabel;
%         %     plabel_PCA(:,:,randi) = reshape(label_PCA,n,m);
%         [OA.PCA(randi),AA.PCA(randi),K.PCA(randi),accurperclass.PCA(:,randi)] = classficationresult(CR2.plabel,testlabels);
%         %% FPCA+SVM
%         %     pcaf = pca_fd(fdobjT, 70);   %函数型PCA
%         %     pcaf = pca_fd(allfdobjT, 70);   %函数型PCA
%         pcaf = pca_fd(allfdobj, 0.9999);   %函数型PCA
%         train_final = pcaf.harmscr(1:numtrain, :);  %训练样本的函数型特征
%         test_final = pcaf.harmscr(numtrain + 1:num, :);   %测试样本的函数型特征
%         %     bg_final = pcaf.harmscr(num + 1:end, :);   %背景样本的函数型特征
%         %svm分类
%         [~, bestc, bestg] = SVMcgForClass(trainlabels,train_final,-10,10,-10,10,10,1,1);
%         cmd3 = ['-c ', num2str(bestc), ' -g ', num2str(bestg), ' -t ', num2str(2)];
%         %     cmd3 = ['-c ', num2str(100), ' -g ', num2str(1000), ' -t ', num2str(2)];
%         model = svmtrain(trainlabels, train_final, cmd3);
%         type = 2;
%         CR3 = ClassResult(testlabels, test_final, model, type);
%         %     bgCR3 = ClassResult(allbglabel, bg_final, model, type);
%         %     accFPCA(randi) = CR3.accuracy(1);  %精度
%         %     label_FPCA = Xlabel;
%         %     label_FPCA(testdataNo) = CR3.plabel;
%         %     label_FPCA(allbgNo) = bgCR3.plabel;
%         %     plabel_FPCA(:,:,randi) = reshape(label_FPCA,n,m);
%         [OA.FPCA(randi),AA.FPCA(randi),K.FPCA(randi),accurperclass.FPCA(:,randi)] = classficationresult(CR3.plabel,testlabels);
%         %% LDA + SVM
%         Ospectraltrain = sample(:,1:numtrain)';
%         Ospectraltest  = sample(:,1+numtrain : num)';
%         
%         Ospatialtrain = sampleZ(:,1:numtrain)';
%         Ospatialtest  = sampleZ(:,1+numtrain : num)';
%         
%         
%         options.Regu = 1;
%         [eigvector,~] = LDA(Ospectraltrain, trainlabels, options);
%         OTrain_data = Ospectraltrain*eigvector;
%         OTest_data = Ospectraltest*eigvector;
%         clear eigvector;
%         
%         [eigvector,~] = LDA(Ospatialtrain, trainlabels, options);
%         OTrain_dataZ = Ospatialtrain*eigvector;
%         OTest_dataZ = Ospatialtest*eigvector;
%         clear eigvector;
%         
%         
%         OTrain_feature = [OTrain_data, OTrain_dataZ];
%         OTest_feature = [OTest_data, OTest_dataZ];
% %         OTrain_feature = [OTrain_data];
% %         OTest_feature = [OTest_data];
%         % svm分类
% %         [~, bestc, bestg] = SVMcgForClass(trainlabels,OTrain_data,-10,10,-10,10,10,1,1);
% %         cmd4 = ['-c ', num2str(bestc), ' -g ', num2str(bestg), ' -t ', num2str(2)];
%         cmd4 = ['-c ', num2str(2), ' -g ', num2str(128), ' -t ', num2str(2)];
%         model = svmtrain(trainlabels, OTrain_feature, cmd4);
%         type = 2;
%         CR4 = ClassResult(testlabels, OTest_feature, model, type);
%         %     accLDA(randi) = CR4.accuracy(1);  %精度
%         %     label_LDA = Xlabel;
%         %     label_LDA(testdataNo) = CR4.plabel;
%         %     label_LDA(allbgNo) = bgCR4.plabel;
%         %     plabel_LDA(:,:,randi) = reshape(label_LDA,n,m);
%         [OA.LDA(randi),AA.LDA(randi),K.LDA(randi),accurperclass.LDA(:,randi)] = classficationresult(CR4.plabel,testlabels);
%         %% FDDA + SVM
%         spectraltrain = coef(:,1:numtrain)';
%         spectraltest  = coef(:,1+numtrain : num)';
%         
%         spatialtrain = coefZ(:,1:numtrain)';
%         spatialtest  = coefZ(:,1+numtrain : num)';
%         
%         options.Regu = 1;
%         [eigvector,~] = LDA(spectraltrain, trainlabels, options);
%         Train_data = spectraltrain*eigvector;
%         Test_data = spectraltest*eigvector;
%         clear eigvector;
%         
%         [eigvector,eigvalue] = LDA(spatialtrain, trainlabels, options);
%         Train_dataZ = spatialtrain*eigvector;
%         Test_dataZ = spatialtest*eigvector;
%         clear eigvector;
%         
% %         Train_feature = [Train_data,Train_dataZ]; Test_feature =
% %         [Test_data, Test_dataZ];
%         
%         Train_feature = [Train_data];
%         Test_feature = [Test_data];
%         
%         % svm分类
%         [bestCVaccuracy, bestc, bestg] = SVMcgForClass(trainlabels,Train_data,-10,10,-10,10,10,1,1);
%         cmd5 = ['-c ', num2str(bestc), ' -g ', num2str(bestg), ' -t ', num2str(2)];
% %         cmd5 = ['-c ', num2str(1), ' -g ', num2str(256), ' -t ',
% %         num2str(2)];
%         model = svmtrain(trainlabels, Train_feature, cmd5);
%         type = 2;
%         CR5 = ClassResult(testlabels, Test_feature, model, type);
%         %     accFDA(randi) = CR5.accuracy(1);  %精度 label_FDA = Xlabel;
%         %     label_FDA(testdataNo) = CR5.plabel; label_FDA(allbgNo) =
%         %     bgCR5.plabel; plabel_FDA(:,:,randi) = reshape(label_FDA,n,m);
%         [OA.FDA(randi),AA.FDA(randi),K.FDA(randi),accurperclass.FDA(:,randi)] = classficationresult(CR5.plabel,testlabels);
        
        clear traindataNo testdataNo
    end
    Result_FSVM(:,11-sstep) = [mean(OA.FSVM);mean(AA.FSVM);mean(K.FSVM)];
%     Result_PCA(:,11-sstep) = [mean(OA.PCA);mean(AA.PCA);mean(K.PCA)];
%     Result_FPCA(:,11-sstep) = [mean(OA.FPCA);mean(AA.FPCA);mean(K.FPCA)];
%     Result_LDA(:,11-sstep) = [mean(OA.LDA);mean(AA.LDA);mean(K.LDA)];
%     Result_FDA(:,11-sstep) = [mean(OA.FDA);mean(AA.FDA);mean(K.FDA)];
end
% toc;
delete(gcp);



