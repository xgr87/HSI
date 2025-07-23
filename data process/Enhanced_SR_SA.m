clc,clear

dir = 'datasets\';
addpath(genpath(pwd))
load([dir,'Salinas.mat']);
load([dir,'Salinas_gt.mat']);

spectral = salinas;
[n,m,dim] = size(spectral);
Xlabel = double(reshape(salinas_gt,n*m,1));

X = reshape(spectral,n*m,dim);
X = mapminmax(X);
X(X==0) = eps;

AfX = fliplr(X);  % 对称延拓
EX = [X, AfX(:,2:end)];
numberoflabel = max(Xlabel);

Onesample = [];
numa = 45;
for i = 1:numberoflabel
    ind = find(Xlabel==i);
    numberofdata(i) = length(ind);
    if(numberofdata(i) ~= 0)
        No{i} = randperm(numberofdata(i));
        Onesample(i,:) = EX(ind(No{i}(1)),:);
        [AFD_fea(i,:),err(i),AFDa(i,:),AFDcoef(i,:)]=AFD(hilbert(Onesample(i,:)),numa);
        err_AFD(i) = norm(real(AFD_fea(i,1:dim))-Onesample(i,1:dim));   %计算误差
    end
end
mean_err_AFD = mean(err_AFD);
%% B-spline
% samples = Onesample(:,1:2*dim-1);
samples = EX;
r = 10^(-3);
rangewavelength = r * [1, 2*dim-1]';
norder = 4;
wavelength = r * (1:2*dim-1)';
nbasis = norder + length(wavelength) - 2;
Lfd = 2;
basisobj = create_bspline_basis(rangewavelength, nbasis, norder, wavelength);
fdnames{1} = 'Wavelength';
fdnames{2} = 'Substance';
fdnames{3} = 'Radiance';

%search the optimal lambda
% lnlam = -8:1:0;
% gcvsave = zeros(length(lnlam),1);
% for i=1:length(lnlam)
%     fdParobj = fdPar(basisobj, Lfd, 10^lnlam(i));
%     [~, ~, gcv] = smooth_basis(wavelength, samples', fdParobj, [], fdnames);   %the default is 'method' =  'chol', but if 'method' = 'qr', the qr decomposition is used.
%     gcvsave(i) = sum(gcv);
% end
% [~, k] = max(-gcvsave);
% lambda = 10^lnlam(k);
lambda = 1e-10;

fdParobj = fdPar(basisobj, Lfd, lambda);
harmbasis = getbasis(getfd(fdParobj));
Jmat_Bp = inprod_basis(harmbasis, basisobj);  %基矩阵

[fdobj, df, gcv, beta, SSE] = smooth_basis(wavelength, samples', fdParobj, [], fdnames);
fdmat = eval_fd(wavelength, fdobj)';    %reconstrcution
coef = getcoef(fdobj)';  %coefficient
err_Bp = sqrt(sum(((samples(:,1:dim) - fdmat(:,1:dim)).^2)'));
mean_err_Bp = mean(err_Bp);
% producing higher spectral resolution
sstep = 5;
wavelength_ex = interp(wavelength, sstep);     %在其中插入一些数据，使得插值后的序列长度为原來的2倍。
fdmatex = eval_fd(wavelength_ex, fdobj)';    %reconstrcution
salinas_ex = fdmatex(:,1:dim*sstep);
salinas_ex(:,1:sstep:end) = X;
salinas = reshape(salinas_ex,n,m,dim*sstep);
save(strcat(dir, "Salinas1120.mat"), "salinas")