function [C, G, cv]=cross_validation_svm(trainlabels,trainsamples)
% This function performs a cross_validation to select goods
% parameter for the training of a binary svm
%
% INPUT
% train_label: the label in column
% train_set: the sample corresponding to the label in row
%
% OUPUT
% C: the optimal value of c coressponding to the best cv
% g: the optimal value of g coressponding to the best cv
% cv: the best cross validation accuracy
% cv_t: the cross validation grid
%written by Zhijing Ye on 10-7-2020

c=2.^(-10:10);
g=2.^(-10:10);

c_s=length(c);
g_s=length(g);

parfor i=1:g_s
    for j=1:c_s
        parameter=sprintf('-c %f -g %f -m 1000 -v 5 -q',c(j),g(i));
        cv_t(i,j)=svmtrain(trainlabels,trainsamples,parameter);
    end
end

[li, co]=find(max(max(cv_t))==cv_t);
C=c(co(1));
G=g(li(1));
cv=max(max(cv_t));
end