function results = mysvm(trainlabels,trainsamples,testlabels,testsamples,trainindex,testindex,labelmap)
% INPUT
% trainlabel and testlabels: the label in column
% trainsamples and testlabels: the sample corresponding to the label, each
% row is a sample.
%
% OUPUT
% results including results.OA, results.AA, results.K, results.accurperclass and
% results.map

[n,m] = size(labelmap);
numtest = length(testlabels);
Ccv = 2^10;
Gcv = 2^-4;
% [Ccv,Gcv]=cross_validation_svm(trainlabels,trainsamples);
parameter=sprintf('-c %f -g %f -t 2',Ccv,Gcv);
model=svmtrain(trainlabels,trainsamples,parameter);
k = floor(numtest/10000);
prelabel = [];

if k < 1
    [prelabel,~,~] = svmpredict(testlabels,testsamples,model);
else
    parfor i = 1 : k
        [plabel,~,~] = svmpredict(testlabels(1+10000*(i-1):10000*i),testsamples(1+10000*(i-1):10000*i,:),model);
        prelabel = [prelabel,plabel];
    end
    [plabel,~,~] = svmpredict(testlabels(1+10000*k:end),testsamples(1+10000*k:end,:),model);
    prelabel = [prelabel;plabel];
    clear plabel;
end

prelabel(trainindex) = trainlabels;
results.map = reshape(prelabel,n,m);
[results.OA, results.AA, results.K, results.accurperclass] = classficationresults(prelabel(testindex),testlabels);
end




