function [Q1, Q2] = getPerformance(x, xhat)
if length(x) == length(xhat)
    Q1 = sum((x - xhat).^2)/len(x)/var(x, xhat);
    Q2 = cov(x, xhat)/sqrt(var(x)*var(xT));
else
    error('vectors not of same size');
end
end