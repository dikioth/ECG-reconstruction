function [out, F] = customKF(xT, x, p)
% customKF computes a custom KALMAN FILTER.
% Input:
%       - xT: Target signal.
%       - x : Input signal.
%       - p : Filter tap.
%
% Output:
%       - xhat
%
% The stated space used is:
%
%      x[n]      [a0 a1 ... ap ]   x[n-1]
%      x[n-1]    [1  0   ... 0 ]   x[n-1]
%      x[n-2] =  [0  1   ... 0 ]   x[n-3]  +   wk
%       ...      [.  .   ... . ]   ...
%      x[n-p]    [0  0   ... 0 ]   x[n-p]
%
%       wher a0 .. ap are computed using the Yule-Walker equation.
%

% Verticalizing vectors
xT = xT(:);
x = x(:);

% Matrix construction.
[ai, ~] = aryule(xT,p-1);
F = vertcat(ai, eye(p-1, p));

H = cat(2, 1, zeros(1, p-1));

% Initiation

numiter = length(xT);
xhatv = zeros(p,1000);
xhat = zeros(p,1);
P = 100 * eye(p);
Q = 0.1; % Process Noise Covariance.
R = 0.1; % Measurement noise covariance.

for n = p:length(xT)
    % Prediction
    xhat = F * xhat;                % The LMMSE prection x
    P = F * P * F' + Q;             % the covariance of LMMSE prediction error of x.
    xhatv(:,n) = xhat;              % Innovation. Difference between model and observation
    
    if n <=71250
    % Update.
    y = xT(n) - H * xhat;           % (4) Covariance of innovation.
    S = H * P * H' + R;             % Optimal Kalman gain.

    K = P * H' / S;                 % LMMSE filter estimate of x.
    xhat = xhat + K * y;            
    P = P - K * H * P;              % MMSE error covariance.
    end
    
end

out = xhatv;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Notes:
% 1. In ARMA model. The autocorelation function (ACF) helps MA order
%    and the partial autocorrelation function (PACF) helps the AR order.
% 2. From lecture notes 4. Equations are: 
%    (1) - The LMMSE prection x, 
%    (2) - the covariance of LMMSE prediction error of x.
%    (3) - Innovation. Difference between model and observation.
%    (4) - Covariance of innovation.
%    (5) - Optimal Kalman gain.
%    (6) - LMMSE filter estimate of x.
%    (7) - MMSE error covariance.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%