function [err,y,h,k]=gall(x,d,M,beta,alpha,epsi,k0,h0)
%
% [err,y,h,k]=gall(x,d,M,beta,alpha,epsi,k,h)
%
% Gradient adaptive Laguerre Lattice Filter. Implements an adaptive
% filter with the option to include an adaptable pole which can be
% useful to model system with long impulse response. Step size is
% optional. The filter operates in two modes: 1) Learning mode
%    and 2)Output (Freeze mode).
%    
% ***Mode 1:The arguments for the Learning mode are: 
%
% x      -measurement data
% d      -desired response or reference signal (same size as x)
% M      -filter order
% beta   -Forgetting factor (0<=beta<=1). beta =1 remembers all the data.
% alpha  -Scalar value of the magnitude of the pole in the filter (0<= alpha <=1).
%         Alpha of 0 is equivalent to the standard classical GAL filter.
%epsi    -small positive constant that initializes the algorithm (epsi <<1)
%k0      -optional 1xM+1 initial reflection coefficients vector
%h0      -optional 1xM+2 initial FIR ladder coefficient vector
%
% Function returns:
%
% err     -learning error curve for the algorithm
% y       -prediction curve (dhat)
% h       -FIR coefficients of the ladder
% k       -Lattice coefficients
%
%
% ***Mode 2:The arguments for the Ouput (Freeze) mode are: 
%
% x      -measurement data
% d      -desired response or reference signal (same size as x)
% M      -empty []
% beta   -[]
% alpha  -Scalar value of the magnitude of the pole in the filter (0<= alpha <=1).
%         Alpha of 0 is equivalent to the standard classical GAL filter.
% k      -Reflection coefficients of the trained lattice.
% h      -Coefficients of the second stage FIR filter.
% Function returns:
%
% err     -Error between filter's output and reference signal
% y       -Filter's output
% h       -[]
% k       -[]
% 
% Reference: Fezjo &  Lev-Ari (1997) IEEE Trans. SP
% 
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
% %%******Example: Robuts Echo cancellation
% %Algorithm performs standard echo cancelation at t<0 and than
% %a sudden cross-talk inteference is added at t=0.
% 
% N=2^12;
% h=[1.0000 -1.8976 1.1870 -0.0642 -0.2718 0.0858];
% h2=[1.0000 -2.0134 1.4571 -0.2602 -0.2377 0.1015];
% fs=20;
% epsi=10^-2;
% w=10;
% t1=-50;
% t=linspace(t1,t1+N/fs,N);
% [trash,N0]=min(abs(t));
% S=zeros(1,N);
% 
% for i=1:20
%     %%%SIGNAL GENERATION
%     %No cross-talk Phase
%     s=sawtooth(t,0.5);
%     n=randn(1,N);
%     d=s+2*n;
%     r=filter(1,h,n*2);
% 
%     %Add Cross-Talk Component to reference signal
%     xtalk=r;
%     q=10;
%     talk=filter(ones(q,1)./q,1,s(N0:end));
%     xtalk(N0:end)=xtalk(N0:end) + talk*std(xtalk)./std(talk);
%     dtalk=d;
% 
%     %%%TECHNIQUE%%%%
%     %Cross Talk Resistance - using Two GAL Filters
%     x1=xtalk(1:N0-1);
%     d1=dtalk(1:N0-1);
%     %Initial filter- Estimates Noise TF and Freeze
%     [err,n_est3,w3,k_xtr]=gall(x1,d1,w,1,0,10^-3,[],[]);
%     [s_est3,n_est3,trash,trash]=gall(xtalk,dtalk,[],[],0,[],k_xtr,w3);
%     s_est3=s_est3(1:end-1,end);
% 
% 
%     %Estimate XTalk Interference
%     [n_est,xtalk_est,w3,k_xtr]=gall(s_est3(N0:end-1,end),n_est3(N0:end),w,0.9975,0.001,10^-3,[],[]);
%     s_est=s_est3';
% 
%     %s_est4(N0:end)=[zeros(1,w-1) dtalk(N0:end-w+1)]' - n_est4(1:end,end);
%     s_est(N0:end)=dtalk(N0:end)' - n_est(1:end,end);
%     s_est(N0:end)=[s_est(N0+w:end) zeros(1,w)];
%     S=s_est./i + (i-1).*S./i;
% end
% 
% %Plot
% figure
% plot(t,S,'r');hold on ;plot(t,s)
% legend('Gall XTR', 'Desired Response')

N=length(x);

if(~isempty(M))
    %Multistage lattice predictor
    f=zeros(1,M+1);
    b=zeros(1,M+1);
    bt=zeros(1,M+1);
    Q=zeros(1,M+1)+epsi;
    delta=zeros(1,M+1);
    y=zeros(N,1);
    D=zeros(1,M+2);
    err=zeros(N+1,M+2);
    b_old=b;
    bt_old=bt;
    if(isempty(k0))
        k=zeros(1,M+1);
    else
        k=k0;
    end
    if(isempty(h0))
        h=zeros(1,M+2);
    else
        h=h0;
    end



    for n=1:N

        f(1)=alpha*b_old(1) + sqrt(1-alpha^2)*x(n);
        b(1)=f(1);
        Q(1) = beta.*Q(1) + b(1).^2;
        err(n,1)=d(n);
        bt_old=bt;
        b_old=b;


        for m=2:M+1

            %Lattice Section
            bt(m-1)= b_old(m-1) + alpha.*(bt_old(m-1) - b(m-1));
            Q(m-1)=beta.*Q(m-1) + (f(m-1).^2 + bt(m-1).^2)./2;
            delta(m)=beta.*delta(m) + f(m-1).*conj(bt(m-1));
            k(m) = delta(m)./Q(m-1);
            f(m)= f(m-1) - k(m)*bt(m-1);
            b(m)= bt(m-1) - conj(k(m))*f(m-1);

        end

        Q(end)=beta.*Q(end) + (f(end).^2 + bt(end).^2)./2;

        for m=1:M+1

            %Ladder Section
            D(m+1)= beta.*D(m+1) + err(n,m).*conj(b(m));
            h(m+1)= D(m+1)./Q(m);
            err(n,m+1)=err(n,m) - h(m+1).*conj(b(m));

        end

        y(n)= h(2:end)*conj(b)';


    end

else

    %Multistage lattice predictor
    k=k0;
    h=h0;
    M=length(k)-1;
    f=zeros(1,M+1);
    b=zeros(1,M+1);
    bt=zeros(1,M+1);
    y=zeros(N,1);
    err=zeros(N+1,M+2);
    b_old=b;
    bt_old=bt;

    
    for n=1:N

        f(1)=alpha*b_old(1) + sqrt(1-alpha^2)*x(n);
        b(1)=f(1);
        err(n,1)=d(n);
        bt_old=bt;
        b_old=b;


        for m=2:M+1

            %Lattice Section
            bt(m-1)= b_old(m-1) + alpha.*(bt_old(m-1) - b(m-1));
            f(m)= f(m-1) - k(m)*bt(m-1);
            b(m)= bt(m-1) - conj(k(m))*f(m-1);

        end
        for m=1:M+1

            %Ladder Section
            err(n,m+1)=err(n,m) - h(m+1).*conj(b(m));

        end
        y(n)= h(2:end)*conj(b)';

    end

    h=[];
    k=[];
end


%err=err(:,end);

