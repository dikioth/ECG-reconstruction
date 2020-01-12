function taps = customRLS(xT, lambda, varargin)
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


% handling reference signals.
if mod(length(varargin), 2)  == 0
    % If varargin comes in pairs, e.g: x1, N.
    numpairs = length(varargin)/2;
    
    for np = numpairs
        sumTaps = sum([varargin{2:2:end}]);
        startIter = max([varargin{2:2:end}]);
    end
else
    error('Reference signal and filter taps should come in pairs.');
end


h = ones(sumTaps, 1);
P = 100 * eye(sumTaps);


thetaA(:, 1) = h;

for n = startIter:length(xT)
    
    Un = [];
    for np = 1:numpairs
        xref = varargin{2*np-1};
        NN = varargin{2*np};
        x = xref(n:-1:n-NN+1);
        Un = vertcat(Un, x);
    end
    
    K = P * Un / (lambda + Un' * P * Un); % Update kalman gain.
    e = xT(n) - h' * Un; % Error from prev. estimate.
    h = h + K * e; % Uppdate filter tap.
    P = 1 / lambda * (eye(sumTaps) - K * Un') * P; % Update inverse
    thetaA(:, n+1) = h; % Storing filter taps into array.
end

taps = thetaA(:, end);


end

% Notes:
% 1. The value of alpha is chosen large to prevent the biasing of the
%    estimator towards the initial estimate