function [out] = customKF2(xT, x, p)
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
%      x[n-2] =  [0  1   ... 0 ]   x[n-3]  +   !TODO: complete this.
%       ...      [.  .   ... . ]   ...
%      x[n-p]    [0  0   ... 0 ]   x[n-p]
%
%
% The Algorithm used is [See Note 2]:
% Prediction:
%         xhat[k|k-1] = F * xhat[k-1|k-1] * G * u[k-1]      (1)
%         P[k|k-1] = F * P[k-1|k-1] * F' + Q                (2)
% 
% Updates:
%         ytilde[k] = y[k] - H * xhat[k|-1]                 (3)
%         S[k] = H * P[k|k-1] * H' + R                      (4)
%         K[k] = P[k|k-1] * H^T * S^-1                      (5)
%         xhat[k|k] = xhat[k|k-1] + K[k] * ytilde           (6)
%         P[k|k] = P[k|k-1] - K[k] * H * P[k|k-1]           (7)

% Verticalizing vectors
xT = xT(:);
x = x(:);

% Matrix construction.
F = diag(xT(1:p));
H = x;

% Initiation

numiter = length(xT);
xhatv = zeros(p,1000);
xhat = zeros(p,1);
P = 100 * eye(p);
Q = 0.1; % Process Noise Covariance.
R = 0.1; % Measurement noise covariance.

for n = 1:length(xT)-p

    % Prediction
    xhat = F * xhat; % + G u[k-1]
    P = F * P * F' + Q; % + Q
    xhatv(:,n) = xhat;

    % Update.
    y = xT(n) - H * xhat;
    S = H * P * H' + R;

    K = P * H' / S;
    xhat = xhat + K * y;
    P = P - K * H * P;
    
    
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