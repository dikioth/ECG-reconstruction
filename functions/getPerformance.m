function [Q1, Q2] = getPerformance(xT_missing, xhat)
if length(xT_missing) == length(xhat)
    %Q1 = mean((xT_missing-xhat).^2)/var(xT_missing)
    %Q2 = cov(x, xhat)/sqrt(var(x)*var(xhat));
    meanxT_missing = mean(xT_missing);
    meanxhat = mean(xhat);
    xmse = 0;
    xcov = 0;
    counter = 1;
    for n = 71251:75000
        xmse = xmse + (xT_missing(counter) - xhat(counter))^2;
        xcov = xcov + (xhat(counter) - meanxhat) * (xT_missing(counter) - meanxT_missing);
        counter = counter + 1;
    end
    xmse = xmse / (counter - 1);
    xcov = xcov / (counter - 1);

    Q1 = 1 - xmse / var(xT_missing);
    Q2 = xcov / sqrt(var(xT_missing)*var(xhat));
else
    error('vectors not of same size');
end
end