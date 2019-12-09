function consts = customRLS(xT, xref, N,  lambda)
% Custom Recursive Least Square Algorithm (RLS):
% Input:
%   - xT: Target signal
%   - xref: Reference signal
%   - lambda: Forgetting factor 0 < lambda < 1
%
% Output:
%   - coefs: ai and bi coefficients.
%
% Remars: 
%   1. It no a priori info of theta is available, a typical iniziation
%      choise is theta[-1] = 0 and P[-1] = alpha * I.
%      where alpha is a large constant ( 100  in this case).
%      and I is the indetity matrix. 
% 
% Author: Elvis Rodas.



% Initiating values (n=0)

% number of iterations

% number of unknowns, length of theta vector
% We assume that x[n] = theta_1 x[n-1] + \theta_2 x[n-2]


% Initilizations: ai = 0.2, bi = 0.2 and n = 0.
mytheta =ones(N,1);
P=100*eye(N);

% keep theta and error in an array to plot later
%eA=zeros(Nsim,1);
thetaA(:,1) = mytheta;
for n=N:length(xT)
    
    h = xref(n:-1:n-N+1);
    
    e = xT(n) - mytheta' * h;
    
    K= P*h/(lambda^n + h'*P*h); 
    
    mytheta = mytheta + K * e;
    
    P = (eye(N)- K*h.')*P;
    
    thetaA(:, n+1) = mytheta;
   % eA(n,1)=e;  
end

consts = thetaA(:,end);


end

% Notes:
% 1. The value of alpha is chosen large to prevent the biasing of the 
%    estimator towards the initial estimate