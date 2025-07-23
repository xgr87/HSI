clc,clear

dir = 'datasets\';
addpath(genpath(pwd))
load([dir,'PaviaU.mat']);
load([dir,'PaviaU_gt.mat']);

spectral = paviaU;
[n,m,dim] = size(spectral);
Xlabel = double(reshape(paviaU_gt,n*m,1));

X = reshape(spectral,n*m,dim);
X = mapminmax(X);
X(X==0) = eps;

AfX = fliplr(X);  % �Գ�����
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
        err_AFD(i) = norm(real(AFD_fea(i,1:dim))-Onesample(i,1:dim));   %�������
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
Jmat_Bp = inprod_basis(harmbasis, basisobj);  %������

[fdobj, df, gcv, beta, SSE] = smooth_basis(wavelength, samples', fdParobj, [], fdnames);
fdmat = eval_fd(wavelength, fdobj)';    %reconstrcution
coef = getcoef(fdobj)';  %coefficient
err_Bp = sqrt(sum(((samples(:,1:dim) - fdmat(:,1:dim)).^2)'));
mean_err_Bp = mean(err_Bp);
% producing higher spectral resolution
sstep = 2;
wavelength_ex = interp(wavelength, sstep);     %�����в���һЩ���ݣ�ʹ�ò�ֵ������г���Ϊԭ���2����
fdmatex = eval_fd(wavelength_ex, fdobj)';    %reconstrcution
paviaU_ex = fdmatex(:,1:dim*sstep);
paviaU_ex(:,1:sstep:end) = X;
paviaU = reshape(paviaU_ex,n,m,dim*sstep);
save(strcat(dir, "PaviaU206.mat"), "paviaU")