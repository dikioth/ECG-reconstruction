function plotResults(xTm, xhat, Q1, Q2)
%plotRessults: Plots a comparision of the last 2s of the true and
% estimated missing part of the target signal.The true missing part
% is presented as solid line and the reconstruction as a dashed line.
% 
%   Input: 
%           - xTm : 3750 x 1 vector containing the true missing part.
%           - xhat: 3750 x x1 vector containing the reconstruction.
%           - Q1  : scalar containing the quality quantity 1.
%           - Q2  : scalar containing quality quantity 2.

figure; hold on; plot(xTm); plot(xhat, '--'); hold off;
xlim([1, 125*2]); % Showing last 2s.
title(sprintf('Reconstruction. Q1 = %.2g, Q2 = %.2g', Q1, Q2));
xlabel('N');
ylabel('Voltage [mV]');
end