function h =  customADAM(xT, varargin)
% customADAM: ADAM optimizer.
% h = customADAM(xT, x1, N, x2*, M*, ...)
% Input:
%       - xT: The target signal.
%       - x1: Reference signal.
%       - N : Filter tap for x1.
%       - x2*: Optional additional reference signal.
%       - M *: Optional additional filter tap to x2.
%
%  OBS: Additional reference signals and filter length must come in pairs.
% Output:
%       - h: The optimal coefficients.
%

% Handle arguments
if mod(length(varargin), 2) == 0
    % if varargin comes in pairs, e.g x1 and N.
    numpairs = length(varargin)/2;
    
    taps = [varargin{2:2:end}];
    xrefs = [varargin{1:2:end}];
    
    sumTaps = sum(taps);
    startIter = max(taps);
else
    help customADAM;
    error('customADAM: Incorrect usage of function');
end


% Default values. Suggested by ref [1]
alpha = 0.001;
beta1 = 0.9;
beta2 = 0.999;
epsilon = 1e-8;

% Initializing
h = ones(sumTaps,1);
m = zeros(sumTaps,1);
v = zeros(sumTaps,1);

y = zeros(sumTaps,1);

for n = startIter:length(xT)
    ii = 0;
    for np = 1:numpairs
        tap = taps(np);
        y(ii+1:tap+ii,1) = xrefs(n:-1:n-tap+1, np);
        ii = tap;
    end
    
    d = xT(n);  % desired signal.
    g = -y' * (d - h'*y);   % gradient.
    m = beta1.* m + (1 - beta1).* g(:);
    v = beta2.* v + (1 - beta2).* g(:).^2;
    mhat = m./ (1 - beta1^n);
    vhat = v./ (1 - beta2^n);
    h = h - alpha.*mhat./(sqrt(vhat) + epsilon);
end
end

% References:
% [1] D. P. Kingma and J. Ba, Adam: A method for stochastic optimization,?
% Latest version available online: https://arxiv.org/abs/1412.6980.