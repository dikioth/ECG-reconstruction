function h =  adamOpt(xT, x1, N, x2, M)
% customADAM: ADAM optimizer.
% h = customADAM(xT, [x1, x2, ...], [N, M, ... ])
% Input:
%       - xT: The target signal.
%       - xref: An N x M matrix, wher M is the num of channels.
%       - N: the filter tap number.
% Output:
%       - h: The coefficients.
%



% Default values. Suggested by ref [1]
alpha = 0.001;
beta1 = 0.9;
beta2 = 0.999;
epsilon = 1e-8;




if nargin == 5
    oneRefSignal = false;
    NM = N+M;
    startIter = max(N,M);
elseif nargin == 3
    oneRefSignal = true;
    NM = N;
    startIter = N;
end
    % Initializing
    h = 1*ones(NM,1);
    m = zeros(NM,1);
    v = zeros(NM,1);

for n = startIter:length(xT)
    
    if oneRefSignal == true
        y = x1(n:-1:n-N+1);
    else
        y = [x1(n:-1:n-N+1); x2(n:-1:n-M+1)];
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
% [1] D. P. Kingma and J. Ba, ?Adam: A method for stochastic optimization,?
% Latest version available online: https://arxiv.org/abs/1412.6980.