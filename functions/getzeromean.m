function [xzm, xmean] = getzeromean(x)
% [xzm, xmean] = getzeromean(x)
% input:
% x - the signal.
% Output:
% xzm  : The zero meaned signal x.
% xmena: the mean of x.

xmean = mean(x);
xzm = x - xmean;
end