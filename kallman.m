function [xhat,XHAT,ALPHA]=kallman(U,d,varargin)
%
%[xhat,XHAT,ALPHA]=kallman(x,d,xhat0,lamb,K0,sig)
%Implements unforced kallman filter. Tap weights are the state vectors
%inputs are C(t)
% Copyright (C) 2010 Ikaro Silva

%Initialize all parameters
[N,M]=size(U);
lamb=0.99;

if (nargin>2 && ~isempty(varargin{1}) )
    xhat=varargin{1};
    xhat=xhat(:);
else
    xhat=rand(M,1)*0.0001;
end
if (nargin>3 && ~isempty(varargin{2}))
    lamb=varargin{2};
end
if (nargin>4 && ~isempty(varargin{3}))
    K=varargin{3};
else
    K=corrmtx(xhat,M-1);
    K=K'*K;
end
if (nargin>5 && ~isempty(varargin{4}))
    sig=varargin{4};
else
    sig=ones(N,M);  %measureement noise (assumed diagonal)!!!!
end


%Do the training
lamb=lamb^(-0.5);
ALPHA=zeros(N,1);
XHAT=zeros(N,M);

for n=1:N
    
    u=U(n,:)';   
    den=diag(1./(u'*K*u + sig(n,:)));
    g= lamb*den*K*u ;
    alpha = d(n) - u'*xhat;
    xhat= lamb *xhat + g*alpha;
    K = (lamb^2)*K - lamb*g*u'*K;
    ALPHA(n)=alpha;
    XHAT(n,:)=xhat';
    
end


