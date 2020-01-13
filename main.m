
%% SIGNAL RECONSTRUCION.

% Description:
% The last 30s of the target signal xT will be reconstructed using
% two reference signals; x1 and x2. Size are
%
%   xT: 71250x1 vector. (missing 3750 will be reconstructed).
%   x1: 75000x1 vector.
%   x2: 75000x1 vector.
%
% Assumming that the target signal is a linear combination of x1 and x2.
%
%           N                     M
%          ===                   ===
%          \                     \
% xT[n] =  /    a_i * x1[n-i] +  /    b_i * x2[n-i]         (*)
%          ===                   ===
%          i = 0                 i = 0
%
% Algorithm used is:
% step 1: Using the frst 9.5 minutes of xT[n], x1[n], x2[n],
%         train the RLS filter and estimate the coefficients ai and bi.
%         Methods used for this steps are: RLS and ADAM optimizer.
%
% step 2: Estimate the last 0.5 minute (125*30 samples) using equation (*)
%         and the coefficients computed in step 1.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% METHOD 1: RLS ALGORITHM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all; clc;

% Importing filters and custom functions files.
addpath('filters', 'functions');

p = 1;          % Patient number
M = 10;         % Filter tap for x1
N = 48;         % Filter tap for x2
lambda = 0.99;  % Forgetting factor

% Loading patient data.
p = getpatient(p);

% Filter coefficients. 'zm' indicates signals with substracted mean.
ci = customRLS(p.xTzm, lambda, p.x1zm, M, p.x2zm, N);

% Reconstructing the last 30 secs (125*30 = 3750 samples)
xhat = getReconstruction(ci, p.x1zm, M, p.x2zm, N) + p.xTmean;

% Performaance analysis. xTm is the true missing part of xT.
[Q1, Q2] = getPerformance(p.xTm, xhat);

% Pltting comparision of reconstruction xhat and true missing singal xm.
plotResults(p.xTm, xhat, Q1, Q2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% METHOD 2: ADAM optimizer
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all; clc;

% Importing filters and custom functions files.
addpath('filters', 'functions');

p = 1;      % Patient number
M = 10;     % Filter tap for x1
N = 48;    % Filter tap for x2

% Loading patient data.
p = getpatient(p);

% Filter coefficients. 'zm' indicates signals with substracted mean.
ci = customADAM(p.xTzm, p.x1zm, M, p.x2zm, N);

% Reconstructing the last 30 secs (125*30 = 3750 samples)
xhat = getReconstruction(ci, p.x1zm, M, p.x2zm, N) + p.xTmean;

% Performaance analysis. xTm is the true missing part of xT.
[Q1, Q2] = getPerformance(p.xTm, xhat);

% Pltting comparision of reconstruction xhat and true missing singal xm.
plotResults(p.xTm, xhat, Q1, Q2);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% LOOP 1: SWEEP FOR INDIVIDUAL X1 and X2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This section does the following:
% 1. Performs a sweep from 5 to 200 w/ intervall of 5 for filter taps N&M.
%    the function 'getSweep' finds the optimal N and M which gives the 
%    highest Q1 and Q2.
% 2. Performs a finer sweep of the neighborhood (length 10) of the optimal 
%    N and M w/ intervall of 1. E.g the intervalls |optimal M| <= 10
%    and | optimal N| <= 10.
% 4. Plots the results found in 3.
% 5. Displays a table of of Q1 for N and M.

close all; clc;
addpath('filters', 'functions');

% CHANGE VALUES BELOW.
pn = 4; % patient number. 
Mintervall = 5:5:200;
Nintervall = 5:5:200;
% CHANGE VALUES ABOVE.

% 1. Performing sweep w/ intervall of 5.
result1 = getSweep(pn, Mintervall, Nintervall);
%
% Displaying results from above.
disp('===============================================');
disp('A sweep from 5 to 200 w/ intervall of 5 yields.');
fprintf('Results for Patient %d:\n ', pn);
fprintf('Max Q1: %.2f corresponds to M: %d, N: %d\n', result1.maxQ1, ...
    result1.maxQ1M, result1.maxQ1N);
fprintf('Max Q2: %.2f corresponds to M: %d, N: %d\n', result1.maxQ2, ...
    result1.maxQ2M, result1.maxQ2N);
disp('===============================================');

% 2. Performing sweep with intervall of 1 near optimal M,N optained above.
pm = 5; % <= Change here. plus minus value. 
neighborhoodM = maxQ1M-pm:maxQ1M+pm; % maxQ1M +- 10;
neighborhoodN = maxQ1N-pm:maxQ1N+pm; % maxQ1N +- 10;
result2 = getSweep(pn, neighborhoodM, neighborhoodN); 

% Displaying results.
disp('===============================================');
fprintf('Sweep of M from %d to %d w/ intervall of 1.\n', maxQ1M -10, maxQ1M+10);
fprintf('Sweep of N from %d to %d w/ intervall of 1.\n', maxQ1N -10, maxQ1N+10);

fprintf('Results for Patient %d:\n ', pn);
fprintf('Max Q1: %.2f corresponds to M: %d, N: %d\n', result2.maxQ1, ...
    result2.maxQ1M, result2.maxQ1N);
fprintf('Max Q2: %.2f corresponds to M: %d, N: %d\n', result2.maxQ2, ...
    result2.maxQ2M, result2.maxQ2N);
disp('===============================================');

% 3. Plotting relevant 
neigLen = 2*pm; % Neighborhood length.
hold on;
for i = 1:neigLen+1
    plot(neighborhoodM, result2.Q1v(i+neigLen*(i-1):i+neigLen*i));
end
hold off;

% 4. Showing table of Q1 for report. Rows are M, Columns are N.
reshapedQ1 = reshape(result2.Q1v, neigLen+1, neigLen+1);

table = zeros(neigLen+2, neigLen+2);
table(2:end,1) = neighborhoodM;
table(1,2:end) = neighborhoodN;
[p,q] = size(reshapedQ1); 
table(end-p+1:end, end-q+1:end) = reshapedQ1; 
disp('Q1 for patient 1 Rows:M, columns:N')
disp(table);


% save results
save(fullfile('Results',['sweep_patient', num2str(pn),'.mat']), 'result1');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Method 2: Kalman filter (Uncompleted)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% The target signal xT is defined as
%
% xT[n] = { xT[n-L], for 0 <= n <= L,       (*)
%         {  0       otherwise
%
% where L is the length of the missing part (125*30 = 3750 in this case).

close all; clear all; clc;
% importing filter and custom functions.
addpath('filters', 'functions');

% Loading data.
p2 = getpatient(2);

% Extendind xT as defined in (*)
xTzm_ext = vertcat(p2.xT, zeros(125*30, 1));

% Kalman filter
[xhat, F] = customKF(xTzm_ext, x2zm, 106);

plot(xhat(1,:));


% Signal modelling
close all;
p = 10000;
[ai, w] = aryule(p2.xT,p-1);

F = vertcat(ai, eye(p-1, p));
plot(F*p2.xT(1:p))
hold on; plot([0;p2.xT(1:p)], '--')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PRE PROCESSING DATA. FOR REPPORT.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all;

addpath('filters', 'functions');
addpath(fullfile('PlotPub', 'lib'));

% last 32s for xT, x1 and x2 for patient 2.
p = getpatient(2);
Fs = 125; % sampling freq.
t = (1:length(p.x1))/Fs;

tlim = [565 575];    
lastsecs = 35;
figure(1); pp1 = Plot(t,[p.xT; zeros(125*30,1)]); xlim(tlim)
figure(2); pp2 = Plot(t,p.x1); xlim(tlim)
figure(3); pp3 = Plot(t,p.x2); xlim(tlim)

pp1.Title = 'Target signal xT';
pp2.Title = 'Correlated reference signal x1';
pp3.Title = 'Correlated refernece signal x2';

for pp = [pp1,pp2,pp3]
    pp.BoxDim = [7.16,3];
pp.LineWidth = 2;
pp.XLabel = 'Time [s]';
pp.YLabel = 'Voltage [mV]';
end

