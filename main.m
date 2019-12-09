% Using the frst 9.5 minutes of xT [n], x1[n], x2[n],
% train the RLS filter and estimate the coefficients ai and bi.

% loading data (sampled at Fs = 125Hz).

% Files are:
%    - ECG_X_AVR
%    - ECG_X_II
%    - ECG_X_II_missing
%    - and ECG_X_V
%
%   where X corresponds to patient number and found in folder 'ECG_X_aYZ'
%
% Folders are: ECG_1_a02, ECG_2_a03, ECG_3_a06, ECG_4_a07, ECG_5_a08
%              ECG_6_a09, ECG_7_a11, ECG_8_a12


% clears
close all; clc;


% Starting w/ patient 2 (by teacher suggestion)
p = 2; % Patient number
M = 4;
N = 20;

% Loading data from directory.
[xT, x1, x2, xTmissing] = getpatient(p);

% Normalizing signals.
xTn = xT - mean(xT);
x1n = x1 - mean(x1);
x2n = x2 - mean(x2);
xTnmissing = xTmissing - mean(xTmissing);

% Plot sinals.
%subplots(xT, xTmissing, x1, x2);
subplots(xTn, xTnmissing, x1n, x2n);

%
%ai = customRLS(xTn, x1n(1:size(xT)), M, 0.99);
bi = customRLS(xTn, x2n(1:size(xT)), N, 0.99);




% Reconstructing the last 30 secs
% 30 secs corresponds to N = 3750 samples.


xhat = zeros(1, 3750);
counter = 1;

for n = length(x2)-length(xTmissing)+1:length(x2)
    %xhat(counter) = sum(ai'*x1n(n-M+1:n));
    xhat(counter) = sum(bi'*x2n(n-N+1:n));
    counter = counter + 1;
end

figure;
hold on;
plot(xTnmissing);
plot(xhat);
hold off;

%
%%

close all;
hold on;

plot(xTn);
plot(-x2n);
hold off;
xlim([0 length(xTnmissing)]);
