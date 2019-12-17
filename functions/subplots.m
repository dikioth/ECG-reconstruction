function subplots(xT, xTmissing, x1, x2)
figure;
subplot(3, 1, 1);
plot(xT);
xlim([length(xT) - length(xTmissing), length(x1)]);
subplot(3, 1, 2);
plot(x1);
xlim([length(xT) - length(xTmissing), length(x1)]);
subplot(3, 1, 3);
plot(x2);
xlim([length(xT) - length(xTmissing), length(x1)]);
end
