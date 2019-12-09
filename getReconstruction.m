function xhat = getReconstruction(coeffs, x, lastN)
% into row vectors.
ci = coeffs(:);
x = x(:);

xhat = zeros(lastN,1);
i = 1;
for n = length(x)-lastN+1:length(x)
    xhat(i) = sum(ci'*x(n-length(coeffs)+1:n));
    i = i + 1;
end
end