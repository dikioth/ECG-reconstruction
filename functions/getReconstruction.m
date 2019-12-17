function xhat = getReconstruction(coeffs, x, lastN)
% into row vectors.
ci = coeffs(:);
x = x(:);

xhat = zeros(lastN, 1);
i = 1;
for n = length(x) - lastN + 1:length(x)
    xhat(i) = ci' * x(n:-1:n-length(coeffs)+1);
    i = i + 1;
end
end