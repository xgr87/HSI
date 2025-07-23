clc,clear

dir = 'datasets\';
addpath(genpath(pwd))
load([dir,'Indian_pines.mat']);
load([dir,'Indian_pines_gt.mat']);

spectral = indian_pines;
[n,m,dim] = size(spectral);
Xlabel = double(reshape(indian_pines_gt,n*m,1));

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
indian_pines_ex = fdmatex(:,1:dim*sstep);
indian_pines_ex(:,1:sstep:end) = X;
indian_pines = reshape(indian_pines_ex,n,m,dim*sstep);
save Indian_pines440.mat indian_pines
clear wavelength_ex fdmatex indian_pines_ex indian_pines
%% Fourier
lambda = 1e-9;
nbasis_fourier =45;
basisobj_fourier = create_fourier_basis(rangewavelength, nbasis_fourier);
fdParobj_fourier = fdPar(basisobj_fourier, Lfd, lambda);
harmbasis_fourier = getbasis(getfd(fdParobj_fourier));
Jmat_fourier = inprod_basis(harmbasis_fourier, basisobj_fourier);  %������

[fdobj_fourier, df, gcv, beta, SSE] = smooth_basis(wavelength, samples', fdParobj_fourier, [], fdnames);
fdmat_fourier = eval_fd(wavelength, fdobj_fourier)';    %reconstrcution
coef_fourier = getcoef(fdobj_fourier)';  %coefficient
err_fourier = sqrt(sum(((samples(:,1:dim)  - fdmat_fourier(:,1:dim) ).^2)'));
mean_err_fourier = mean(err_fourier);
%% plot curves
for i = 1 : numberoflabel
    plot(Onesample(i,1:dim),'k--'), hold on, axis([0 200 -1.1 1.1]);
    plot(fdmat_fourier(i,1:dim),'g-'),
    plot(fdmat(i,1:dim),'b-'),
    plot(real(AFD_fea(i,1:dim)),'r-'),hold off;
    %     legend('Original spectral signal','spectral curve by FD','spectral curve by AFD');
    legend('Original spectral signal','spectral curve by FD','spectral curve by B-spline','spectral curve by AFD');
    iptsetpref('ImshowBorder','tight')
    iptsetpref('ImtoolInitialMagnification','fit')
end

f1=figure; 
curveind = [2 6 11 16];
sub_row = 2; % ��ͼ����
sub_col = 2; % ��ͼ����
for i_row = 1 : sub_row
    for j_col = 1 : sub_col
        i = (i_row-1)*sub_col+j_col; % ��ͼ��˳��
        subplot(sub_row, sub_col, i);
        plot(Onesample(curveind(i),1:dim),'k--'), hold on, axis([0 200 -1.1 1.1]);
        plot(fdmat_fourier(curveind(i),1:dim),'g-'),
        plot(fdmat(curveind(i),1:dim),'b-'),
        plot(real(AFD_fea(curveind(i),1:dim)),'r-'),hold off;
        legend('Original spectral signal','Spectral curve by Fourier','Spectral curve by B-spline','Spectral curve by TM');
%         legend('ԭ������Ϣ','��������\_Fourier','��������\_B-Spline','��������\_TM');
        RemoveSubplotWhiteArea(gca, sub_row, sub_col, i_row, j_col); % ȥ���հײ���
        set(gcf, 'PaperPositionMode', 'auto'); % ʹprint����������Ļ��ʾ��С��ͬ
        % suptitle('Indian Pines')
    end
end
% % magnify(f1)

% for i = 1 : 40
%     [AFD_fea,err,AFDa,AFDcoef]=AFD(hilbert(Onesample(2,:)),5*i);
%     err_AFD(i) = norm(real(AFD_fea(1:dim))-Onesample(2,1:dim));   %������
%     
%     nbasis_fourier = 5*i;
%     if nbasis_fourier >= dim
%         nbasis_fourier = nbasis_fourier -1;
%     end
%     basisobj_fourier = create_fourier_basis(rangewavelength, nbasis_fourier);
%     fdParobj_fourier = fdPar(basisobj_fourier, Lfd, lambda);
%     harmbasis_fourier = getbasis(getfd(fdParobj_fourier));
%     Jmat_fourier = inprod_basis(harmbasis_fourier, basisobj_fourier);  %������
%     
%     [fdobj_fourier, df, gcv, beta, SSE] = smooth_basis(wavelength, Onesample(2,:)', fdParobj_fourier, [], fdnames);
%     fdmat_fourier = eval_fd(wavelength, fdobj_fourier)';    %reconstrcution
%     coef_fourier = getcoef(fdobj_fourier)';  %coefficient
%     err_fourier(i) = sqrt(sum(((Onesample(2,1:dim)  - fdmat_fourier(1:dim) ).^2)'));
% end
%% error vs number of basis functions
% xt = 5*(1:40);
% plot(xt,err_fourier,'g-',xt,err_AFD,'r-',202,err_Bp(2),'bo','MarkerFaceColor','b');hold on
% plot([xt,205],err_Bp(2)*ones(1,41),'b--',[45 45],[0 3.5],'k--');
% plot(45,err_AFD(9),'ro','MarkerFaceColor','r');
% xlabel('Number of basis functions');
% ylabel('Error');
% axis([0 205 0 3.5])
% set(gca,'xtick',0:5:205);
% legend('Fourier','TM','B-spline');
% hold off;
% set(gca,'looseInset',[0 0 0 0])
% iptsetpref('ImshowBorder','tight')
% iptsetpref('ImtoolInitialMagnification','fit')


function [] = RemoveSubplotWhiteArea(gca, sub_row, sub_col, current_row, current_col)
% ����OuterPosition
sub_axes_x = current_col*1/sub_col - 1/sub_col;
sub_axes_y = 1-current_row*1/sub_row; % y�Ǵ������µ�
sub_axes_w = 1/sub_col;
sub_axes_h = 1/sub_row;
set(gca, 'OuterPosition', [sub_axes_x, sub_axes_y, sub_axes_w, sub_axes_h]); % ����OuterPosition
end


% save result2.mat samples AFD_fea err_AFD AFDa AFDcoef mean_err_AFD fdmat err_Bp coef Jmat_Bp mean_err_Bp fdmat_fourier err_fourier coef_fourier mean_err_fourier