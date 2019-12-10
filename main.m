% Using the frst 9.5 minutes of xT [n], x1[n], x2[n],
% train the RLS filter and estimate the coefficients ai and bi.

% loading data (sampled at Fs = 125Hz).

% Files are:
%	Variable | Description         | File Name
%   ------------------------------------------------------
%     xT       Target signal.        ECG_X_AVR.mat          
%     x1       Reference signal 1.   ECG_X_II.mat
%     x2       Reference signal 2.   ECG_X_V.mat
%     xTm      Missing part.         ECG_X_II_missing.mat
%
%   where X corresponds to patient number and found in folder 'ECG_X_aYZ'
%   Folders are: ECG_ ...
%         [1_a02 | 2_a03 | 3_a06  | 4_a07 | 5_a08 |6_a09 | 7_a11 |8_a12]



% clears
close all; clc;

% Starting w/ patient 2 (by teacher suggestion)
p = 8; % Patient number
M = 5; % Filter tap for x1
N = 105; % Filter tap for x2

% Loading data from directory.
[xT, x1, x2, xTm] = getpatient(p);

% Normalizing signals.
xTn = xT - mean(xT);
x1n = x1 - mean(x1);
x2n = x2 - mean(x2);
xTmn = xTm - mean(xTm);

% % Filter coefficients theta for x1 and x2.
% ci = customRLSh(xT, x1, x2, M, N);
% ai = ci(1:M);
% bi = ci(M+1:end);
% 
% 
% % Reconstructing the last 30 secs (3750 samples)
% xhatn1 = getReconstruction(ai,x1n, 3750);
% xhatn2 = getReconstruction(bi,x2n, 3750);
% 
% xhat = xhatn1 + xhatn2 + mean(xT);
% 
% % Performaance analysis
% [Q1, Q2] = getPerformance(xTm, xhat);
% 
% % Plots
% 
% % Comparing reconstruction.
% figure;
% hold on;
% plot(xTm);
% plot(xhat);
% hold off;
% 
% % KALLMAN
% close all;

[a,b,c] = kallman([x1n(1:length(xT)), x2n(1:length(xT))], xT);
xRec1 = getReconstruction(a(1), x1n, 3750);
xRec2 = getReconstruction(a(2), x2n, 3750);
xRec = xRec1 + xRec2 + mean(xT);
subplot(2,1,1);
plot(xRec);
subplot(2,1,2);
plot(xTm);


figure;
plot([xTm(end-375:end), xRec(end-375:end)]);


ylabel('Voltage [mV]');
xlabel('N');

legend('ECG II', 'Reconstructed ECG II');




