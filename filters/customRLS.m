function taps = customRLS(xT, xref, N, lambda)
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


h = ones(N, 1);
P = 100 * eye(N);


thetaA(:, 1) = h;

for n = N:length(xT)
    Un = xref(n:-1:n-N+1); % next N inputs.
    K = P * Un / (lambda + Un' * P * Un); % Update kalman gain.
    e = xT(n) - h' * Un; % Error from prev. estimate.
    h = h + K * e; % Uppdate filter tap.
    P = 1 / lambda * (eye(N) - K * Un') * P; % Update inverse
    thetaA(:, n+1) = h; % Storing filter taps into array.
end

taps = thetaA(:, end);


end

% Notes:
% 1. The value of alpha is chosen large to prevent the biasing of the
%    estimator towards the initial estimate