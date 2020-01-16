function [sig,varargout]=mcaf(x,varargin)
%
%
%[sig,[mse],[xhat],[param]]=mcaf(x,d_ind,[CH],[Fs],[show],[verbose])
%
%Input Arguments are:
% x       -NxM data matrix. Each column represent a channel
% d_din   -scalar index indicating the target column (to be predicted)
% CH      -Optional Mx1 cell array with the names of each channel 
% Fs      -Optional scalar indicating sampling frequency in Hz (default is 125 Hz)
% show    -Optional logical argument indicating to plot final results
% verbose -Optional logical argument indicating progress of the code
% 
% %Output Arguments are:
% sig     -Nx1 reconstructed signal
% mse     -Optional output Nx1 mean square error of the final estimate and desired response
% xhat    -Optional output 1xM final Kalman weights 
% param   -Optional detailed information of filter training 
%          including final filter parameters and NxM Kalman weights as a function of time
%
%Multi Channel Adaptive Filtering
%Based on the Physionet 2010 Challenge phys_est_phase2_v11 version.
% Copyright (C) 2010 Ikaro Silva
% 
% This library is free software; you can redistribute it and/or modify it under
% the terms of the GNU Library General Public License as published by the Free
% Software Foundation; either version 2 of the License, or (at your option) any
% later version.
% 
% This library is distributed in the hope that it will be useful, but WITHOUT ANY
% WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
% PARTICULAR PURPOSE.  See the GNU Library General Public License for more
% details.
% 
% You should have received a copy of the GNU Library General Public License along
% with this library; if not, write to the Free Software Foundation, Inc., 59
% Temple Place - Suite 330, Boston, MA 02111-1307, USA.
% 
% You may contact the author by e-mail (ikaro@ieee.org).
%
%NOTE: this function uses PARLOOPS, so make sure to type 'matlabpool'
%prior to running it.
%%Example
% 
% clear all;clc;close all
% load mat_c05
% d_ind=2;
% CH={'RESP','PLETH','ABP','II','V','AVR','CVP'};
% verbose=1;
% show=1;
% [sig]=mcaf(mat_c05,d_ind,CH,[],show,verbose);



Fs=125;
pole_order=20;
d_ind=4;
CH={};
par_mode=1;
show=1;
verbose=1;
par_name={'d_ind','CH','Fs','show','verbose'};

%Get optional parameters
for n=2:nargin
    if(~isempty(varargin{n-1}))
        eval([par_name{n-1} '=varargin{n-1};'])
    end
end

if(verbose)
    display('***Running MCAF***');
end
%targets=dlmread([data_dir fname '.missing']);
Ntarget=round(Fs*30);
[N,L]=size(x);
if(isempty(CH))
    CH=cell(L,1);
end
if(length(CH) ~= L)
    error(['CH is ' num2str(length(CH)) ' long, should be a cell array of size ' num2str(L)])
end

%%% GET SIGNALS %%%%
d=x(:,d_ind);
sig_name=CH{d_ind};

%Replace original signal with a estimate of it base on past history
sig=[d(end-2*Ntarget+1:end-Ntarget);d(1:end-Ntarget)];
x(:,d_ind)=sig;

%Get Rid of any initial clipping
clp=cumsum(abs(sign(diff(d))));
clp_end=(find(clp~=0));
if(~isempty(clp_end) && ~clp(Fs) && (clp_end(1) < N-2*Ntarget) )
    d(1:clp_end-1)=[];
    x(1:clp_end-1,:)=[];
    N=length(d);
    warning('Getting rid of initial flat signal artifact');
end

n2=N-Ntarget;
n1=1;
sig=zeros(N,1);
mse=0;
NCh=length(CH);
param.CH=CH;
param.Ntarget=Ntarget;
param.w=NaN;
param.XHAT=NaN;
param.fin_best=' ';
param.init=' ';
param.CH=CH;
param.Ntarget=Ntarget;
param.w=NaN;
param.XHAT=NaN;
param.fin_best=' ';
param.init=' ';
param.n1=n1;
param.n2=n2;
maxy=max(d(1:N-Ntarget));
miny=min(d(1:N-Ntarget));
y=[1:L];
ws1=1;      %sum sqrt
ws2=1;      %rms


%Define GALL Parameters
M=35;
epsi=10^-2;
R=zeros(L,1);
H=zeros(L,M+2);
K=zeros(L,M+1);
D=zeros(n2-n1+1,L);     %estimates for each signal
P=zeros(L,1);
B=zeros(L,1);
G=zeros(L,1); %gain
param.D=D;




%%%%%%%%       ANALYZE THE SIGNALS  %%%%%%%%%%%%%%%%%%%%
%Return DC of input if the signal is flat
if(length(unique(x(1:N-Ntarget,d_ind)))==1)
    sig=ones(N,1).*x(1,d_ind);
    warning('Signal is flat. Returning DC as estimate')
    if(nargout >1)
        varargout(1)={NaN};
        if(nargout>2)
            varargout(2)={NaN};
            if(nargout>3)
                varargout(3)={zeros(1,L)};
            end
        end
    end
    return
end
%%%%%%%%%%%%%%%%%%
%Training Phase
%%%%%%%%%%%%%%%%%%

%Optimize xcross each signal
POLE=[[1-0.0005.^[linspace(0,1,pole_order)]] ];
pad=d(n1:n2).*0;
target=d(n1:n2);
BETA=[[1-0.0001.^[linspace(0,1,6)]] ];
BETA(1)=0.5;
err=zeros(1,L);
std_tar=std(target);
NBETA=length(BETA);
NPOLE=length(POLE);
tmp_err1=zeros(NBETA,NPOLE)+NaN;
tmp_err2=zeros(NBETA,NPOLE)+NaN;
for m=1:L
    if(verbose)
        display(['Getting reconstruction of CH ' num2str(m) ' ( out of ' num2str(L) ' CHs)']);
    end
      for beta=1:NBETA
        parfor p=1:NPOLE
            [trash,dhat_tmp,h,k]=gall(x(n1:n2,y(m)),target,M,BETA(beta),POLE(p),epsi,[],[]);
            [trash,dhat_tmp,trash,trash]=gall(x(n1:n2,y(m)),pad,[],[],POLE(p),epsi,k,h); %test phase
            %tmp_err=sqrt(mean((dhat_tmp(end-8*Ntarget:end)-target(end-8*Ntarget:end)).^2));
            if(any(isnan(dhat_tmp)) || isnan(sum(diff(dhat_tmp))) ||~sum(diff(dhat_tmp)))
                dhat_tmp=x(n1:n2,y(m));
            end
            tmp_err1(beta,p)=sum((dhat_tmp-target).^2);
            tmp_err2(beta,p)=sqrt(mean((dhat_tmp-target).^2));
        end
    end %of optimization loop
    [best_err1,best_ind1]=nanmin(tmp_err1(:));
    [best_err2,best_ind2]=nanmin(tmp_err2(:));
    
    df1=tmp_err1(best_ind1)-tmp_err1(best_ind2);
    df2=tmp_err2(best_ind2)-tmp_err2(best_ind1);
    if((ws1*df1)<(ws2*df2))
        [best_err,best_ind]=nanmin(tmp_err1(:));
    else
        [best_err,best_ind]=nanmin(tmp_err2(:));
    end
    [beta_opt,p_opt]=ind2sub([NBETA NPOLE],best_ind);
    [trash,dhat_tmp,h,k]=gall(x(n1:n2,y(m)),target,M,BETA(beta_opt),POLE(p_opt),epsi,[],[]);
    [trash,dhat_tmp,trash,trash]=gall(x(n1:n2,y(m)),pad,[],[],POLE(p_opt),epsi,k,h); %test phase
     if(any(isnan(dhat_tmp)) || isnan(sum(diff(dhat_tmp))) ||~sum(diff(dhat_tmp)))
          dhat_tmp=x(n1:n2,y(m));
     end
    err(m)=best_err;
    dhat=dhat_tmp;
    H(m,:)=h;
    K(m,:)=k;
    P(m)=POLE(p_opt);
    B(m)=BETA(beta_opt);
    G(m)=std_tar/std(dhat);
    if(isinf(G(m)) || isnan(G(m)))
        G(m)=1;
    end
    D(:,m)=dhat'.*G(m);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Combine each estimator using Kalman filter to obtain final estimate
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bad=find(isnan(err)==1);
if(~isempty(bad))
    warning('NaN error found')
    D(:,bad)=-D(:,d_ind);
end

err(bad)=inf;
%sort the signals according to the error
best_ind=sortrows([err(:) [1:L]']);

%Initiliaze state vector
w0=ones(L,1)./L;
Klamb=[0 0.75 0.9 0.97 0.997 0.9997 0.99997 0.999997 0.9999997 0.99999997];
NKlamb=length(Klamb);
K0=[];
NOISE=[];

k_opt=[inf NaN];
wopt=NaN;
XHAT=NaN;
XHATopt=NaN;
max_warn=0;
Opt_st=10;
mse1=zeros(NKlamb,1)+NaN;
mse2=zeros(NKlamb,1)+NaN;
if(verbose)
    display(['Combining individual reconstructions']);
end
parfor k=1:NKlamb
    k_tmp=Klamb(k);
    [wtmp,XHAT,ALPHA]=kallman(D,target,w0,k_tmp,K0,NOISE);
    dhat_tmp=D*wtmp;
    mse1(k)=sum((dhat_tmp(end-Opt_st:end)-target(end-Opt_st:end)).^2);
    mse2(k)=sqrt(mean((dhat_tmp(end-Opt_st:end)-target(end-Opt_st:end)).^2));
end

[K_err1,best_k1]=nanmin(mse1);
[K_err2,best_k2]=nanmin(mse2);
df1=mse1(best_k1)-mse1(best_k2);
df2=mse2(best_k2)-mse2(best_k1);
if((ws1*df1)<(ws2*df2))
    [K_err,best_k]=nanmin(mse1(:));
else
    [K_err,best_k]=nanmin(mse2(:));
end


[wtmp,XHAT,ALPHA]=kallman(D,target,w0,Klamb(best_k),K0,NOISE);
k_opt(1)=K_err;
k_opt(2)=Klamb(best_k);
wopt=wtmp;
XHATopt=XHAT;
[trash,Max_ind]=max(XHATopt(end-(60*2)*Fs:end,:),[],2);
if(length(unique(Max_ind))>2) %did not have the unique statement before
    max_warn=1;
end
if(max_warn)
    warning('Attempting to stabilize weights')
    mse1_best=[Inf NaN];
    mse2_best=[Inf NaN];
    wopt2=wtmp;
    wopt1=wtmp;
    XHATopt1=[];
    XHATopt2=[];
    Kwin_max=15*Fs;
    for k=1:length(Klamb)
        k_tmp=Klamb(k);
        [wtmp,XHAT,ALPHA]=kallman(D,target,w0,k_tmp,K0,NOISE);        
        %Apply averaging to XHAT  
        for kal_win=[(5*Fs):(5*Fs):Kwin_max]
            wtmp=mean(XHAT(end-kal_win:end,:))';
            dhat_tmp=D*wtmp;
            mse1=sum((dhat_tmp(end-Opt_st:end)-target(end-Opt_st:end)).^2);
            %mse1=mean(abs(dhat_tmp(end-Opt_st:end)-target(end-Opt_st:end)));
            mse2=sqrt(mean((dhat_tmp(end-Opt_st:end)-target(end-Opt_st:end)).^2));
            if(mse1 < mse1_best(1))
                wopt1=wtmp;
                XHATopt1=XHAT;
                mse1_best(1)=mse1;
                mse2_best(2)=mse2;
            end
            if(mse2 < mse2_best(1))
                wopt2=wtmp;
                XHATopt2=XHAT;
                mse1_best(2)=mse1;
                mse2_best(1)=mse2;
            end
        end
    end
    if(ws1*(mse1_best(1)-mse1_best(2)) < ws2*(mse2_best(1)-mse2_best(2)))
        wopt=wopt1;
        XHATopt=XHAT;
        mse_best=mse1;
    else
        wopt=wopt2;
        XHATopt=XHAT;
        mse_best=mse2;
    end
end
if(any(isnan(wopt)))
    warning('Kallman weightst not found.')
    wopt=ones(L,1)./L;
end
if(min(wopt)==0)
    warning('Weights normalized')
    wopt=ones(L,1)./L;
end

%Real Final Deal
D2=zeros(N,L);     %estimates for each signal
parfor m=1:L
    [err_tmp,dhat2,trash,trash]=gall(x(:,y(m)),d.*0,...
        [],[],P(m),epsi,K(m,:),H(m,:)); %test phase
    if(any(isnan(H(m,:))) || ~sum(H(m,:)) || any(isnan(dhat2)) || ~sum(diff(dhat2)))
        dhat2=x(:,y(m));
    end
    D2(:,m)=dhat2'.*G(m);
end
dhat2_A=D2*wopt;

%Update dc component
dhat2_B=dhat2_A-mean(dhat2_A)+mean(d(n2-3*Ntarget:n2));
mseA1=sum((dhat2_A(end-2*Ntarget:end-Ntarget)-d(end-2*Ntarget:end-Ntarget)).^2);
mseA2=sqrt(mean((dhat2_A(end-2*Ntarget:end-Ntarget)-d(end-2*Ntarget:end-Ntarget)).^2));
mseB1=sum((dhat2_B(end-2*Ntarget:end-Ntarget)-d(end-2*Ntarget:end-Ntarget)).^2);
mseB2=sqrt(mean((dhat2_B(end-2*Ntarget:end-Ntarget)-d(end-2*Ntarget:end-Ntarget)).^2));
df1=min(mseA1,mseB1)- (mseA1*(mseA2<mseB2)) - (mseB1*(mseA2>mseB2));
df2=min(mseA2,mseB2)- (mseA2*(mseA1<mseB1)) - (mseB2*(mseA1>mseB1));

if ((ws1*df1) < (ws2*df2))
    if(mseA1<=mseB1)
        dhat2=dhat2_A;
    else
        dhat2=dhat2_B;
    end
else
    if(mseA2<=mseB2)
        dhat2=dhat2_A;
    else
        dhat2=dhat2_B;
    end
end

%Pass final signal through a GALL filter
%Optimize xcross each signal
BETA2=[0 0.5 0.95 0.995 0.9995 0.99995 0.999995 0.9999997 ];
NBETA2=length(BETA2);
opt=inf;
dhat3_tmp=inf;
H3=[];
K3=[];
B3=[];
dhat3=dhat2;
st_ind=max(N-6*Ntarget,1);
tmp_err1=zeros(NBETA2,1)+NaN;
tmp_err2=zeros(NBETA2,1)+NaN;
parfor beta=1:NBETA2
    [trash,trash,h,k]=gall(dhat2(n1:n2),target,M,BETA2(beta),0,epsi,[],[]);
    [trash,dhat3_tmp,trash,trash]=gall(dhat2,d.*0,[],[],0,epsi,k,h); %test phase
    
    tmp_err1(beta)=sum((dhat3_tmp(st_ind:end-Ntarget)-d(st_ind:end-Ntarget)).^2);
    tmp_err2(beta)=sqrt(mean((dhat3_tmp(st_ind:end-Ntarget)-d(st_ind:end-Ntarget)).^2));
    
end %of optimization loop
[beta1_err,best_beta1]=nanmin(tmp_err1(:));
[beta2_err,best_beta2]=nanmin(tmp_err2(:));
df1=tmp_err1(best_beta1)-tmp_err1(best_beta2);
df2=tmp_err2(best_beta2)-tmp_err2(best_beta1);
if((ws1*df1)<(ws2*df2))
    [beta2_err,best_beta2]=nanmin(tmp_err1(:));
else
    [beta2_err,best_beta2]=nanmin(tmp_err2(:));
end   
[trash,trash,h,k]=gall(dhat2(n1:n2),target,M,BETA2(best_beta2),0,epsi,[],[]);
[trash,dhat3_tmp,trash,trash]=gall(dhat2,d.*0,[],[],0,epsi,k,h); %test phase
opt=beta2_err;
dhat3=dhat3_tmp;
H3=h;
K3=k;
B3=BETA2(best_beta2);
dhat3(dhat3>maxy)=maxy;
dhat3(dhat3<miny)=miny;

if(isnan(dhat3))
    dc_tmp=unique(d(1:end-Ntarget));
    if(length(unique(dc_tmp))==1)
        dhat3=zeros(size(d))+dc_tmp;
    end
end
mse=sum((dhat3(n1:n2)-target).^2);
NCh=L;
[trash,fin_best]=max(XHAT(end,:));
[trash,init_best]=max(XHAT(1,:));
[trash,ave_best]=max(mean(XHAT,1));

if(show)
    plot_res(d,dhat3,n2,miny,maxy,CH,d_ind,mse,fin_best,ave_best,XHATopt,y,sig_name,N,init_best);
end
sig=dhat3;
param.R=R;
param.H=H;
param.K=K;
param.d=d;
param.D=D2;
param.P=P;
param.res=(d(1:n2)-dhat3(1:n2)).^2;
param.res_ref=(D(:,d_ind)-d(1:n2)).^2;
param.w=wopt;
param.XHAT=XHATopt;
param.fin_best=CH{fin_best};
param.init=CH{init_best};
param.sqi=1./(1+param.res./var(sig-mean(sig)));

if(nargout >1)
    varargout(1)={mse};
    if(nargout>2)
        varargout(2)={mean(XHAT,1)};
        if(nargout>3)
            varargout(3)={param};
        end
    end
end

if(verbose)
    display('->DONE running MCAF');
end

%%%%%%%%%%%% END OF MAIN %%%%%%%%%%%%%%%%%%%%%%

function plot_res(d,dhat3,n2,miny,maxy,CH,d_ind,mse,fin_best,ave_best,XHATopt,y,sig_name,N,init_best)

%Plot Kallman Results
figure
subplot(311)
plot(d)
hold on;grid on
plot(dhat3,'r')
line([n2 n2],[miny maxy].*1.5,'Color','g','LineStyle','--','LineWidth',3)
if(~isempty(CH{1}))
legend(CH{d_ind},'Estimated')
title(['CH= ' sig_name ' MSE= ' num2str(mse) ,' Final Best= ' CH{fin_best}])
end
xlim([0 N])

subplot(312)
plot(XHATopt)
grid on
if(~isempty(CH{1}))
    legend(CH(y))
    title(['Ave Best Channel: ' CH{ave_best}])
end
xlim([0 N])

subplot(313)
plot((d(1:n2)-dhat3(1:n2)).^2)
xlim([0 N])
