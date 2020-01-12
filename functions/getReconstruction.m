function xhat = getReconstruction(coeffs, varargin)
%getReconstruction: Reconstruct the last 30s of the target signal.
% using the signals x1 and/or x2 with it's filter optimal coefficients.
%
% getReconstruction(coeffs, x1, x2*, x3*, ... )
% 
%   Input:
%       - coeffs: NM x 1 vector containing coefficients of the reference
%                 signals x1 and/or x2. 
%       - x1: Nx1 vector containing reference signal.
%       - x2*: Mx1 vector containing additional reference signals.
%
%       OBS: length(NM) == length(x1) + length(x2) + ... length(xn).
%
%   Output: 
%       - xhat : 3750 x1 vector containing the reconstruction of the last
%                30s of the target signal.


if mod(length(varargin), 2)  == 0
    % If varargin comes in pairs, e.g: x1 with its filter length N.
    numpairs = length(varargin)/2;
    lenxref = length([varargin{1}]);
   
else
    error('Reference signal and filter taps should come in pairs.');
end

lastN = 125*30; % 30 secs
xhat = zeros(lastN,1);

i = 1;
for n = lenxref-lastN+1:lenxref
    ii = 0;
    for np = 1:numpairs
        NN = varargin{2*np};
        c = coeffs(ii + 1:ii + NN);
        xref = varargin{2*np-1};
        xhat(i) = xhat(i) + c'*xref(n:-1:n - NN + 1);
        ii = ii + varargin{2*np};
    end
    i = i+1;
end
end