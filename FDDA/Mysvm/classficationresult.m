function [OA,AA,K,accurperclass] = classficationresult(testlabel,truelabel)

n = length(truelabel);
labelset = unique(truelabel);
classnumber = length(labelset);
classmatrix = zeros(classnumber,classnumber);

for i = 1 : classnumber
    pos = find(truelabel == labelset(i));
    midlabel = testlabel(pos);
    [num,val] = hist(midlabel,labelset);
    classmatrix(i,:) = num;
end
diagelement = diag(classmatrix);
sumdiag = sum(diagelement);
OA = sumdiag/n*100;

accurperclass = diagelement./sum(classmatrix,2)*100;
AA = mean(accurperclass);

Rcsum = sum(sum(classmatrix,1).*sum(classmatrix,2)');
K = (n*sumdiag - Rcsum)/(n*n-Rcsum);