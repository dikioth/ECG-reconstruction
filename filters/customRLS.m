function taps = customRLS(xT, lambda, varargin)
% Custom Recursive Least Square Algorithm (RLS):
% Use:
%       taps = customRLS(xT, lambda, x1, M, X2*, N*, ... )
%
% Input:
%       - xT: Target signal
%       - lambda: Forgetting factor 0 < lambda < 1
%       - x1: Reference signal
%       - M : Filter taps for reference siganal.
% Output:
%       - coefs: ai and bi coefficients.

% handling reference signals from 'varargin'.
if mod(length(varargin), 2)  == 0 % If varargin comes in pairs, e.g: x1, N.
    numpairs = length(varargin)/2; % Store number of pairs.
    
    taps = [varargin{2:2:end}];
    xrefs = [varargin{1:2:end}];
    
    sumTaps = sum(taps);
    startIter = max(taps);
else
    help customRLS;
    error('customRLS: Incorrect usage of function');
end

% Initial values. See note 1 and 2 below.
h = zeros(sumTaps, 1);
P = 100 * eye(sumTaps);

% Uncomment below for storing the filter taps of each iteration.
% thetaA(:, 1) = h; 

Un = zeros(sumTaps,1);
for n = startIter:length(xT)
    
    ii = 0;
    for np = 1:numpairs
        tap = taps(np);
        Un(ii+1:tap+ii,1) = xrefs(n:-1:n-tap+1, np);
        ii = tap;
    end
        
    K = P * Un / (lambda + Un' * P * Un);           % Update kalman gain.
    e = xT(n) - h' * Un;                            % Err. from prev. estimate.
    h = h + K * e;                                  % Uppdate filter tap.
    P = 1 / lambda * (eye(sumTaps) - K * Un') * P;  % Update inverse
    
    % Uncomment below for storing the filter taps of each iteration.
    % thetaA(:, n+1) = h; % Storing filter taps into array.
end

% Uncomment below for storing the filter taps of each iteration.
% taps = thetaA(:, end);
taps = h;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Notes:
%   1. It no a priori info of theta is available, a typical iniziation
%      choise is theta[-1] = 0 and P[-1] = alpha * I.
%      where alpha is a large constant ( 100  in this case).
%      and I is the indetity matrix. This algorithm uses those initial
%      values.
%   2. The value of alpha is chosen large to prevent the biasing of the
%       estimator towards the initial estimate
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%