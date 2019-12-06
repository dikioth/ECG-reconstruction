% Using the ?rst 9.5 minutes of xT [n], x1[n], x2[n],
% train the RLS ?lter and estimate the coefficients ai and bi.

% loading data

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
close all; clear all; clc;

% Starting w/ patient 2 (by teacher suggestion)
p = 2; % Patient number

% Loading data from directory.
listdir = dir;
strAVR = ['ECG_', num2str(p),'_AVR.mat'];
strII = ['ECG_', num2str(p), '_II.mat'];
strII_missing =  ['ECG_', num2str(p), '_II_missing.mat'];
strV =  ['ECG_', num2str(p), '_V.mat'];


for i = 1:length(listdir)
    if strfind(listdir(i).name, ['ECG_', num2str(p)]) == 1
        x2 = importdata(fullfile(pwd, listdir(i).name, strAVR ));
        xT = importdata(fullfile(pwd, listdir(i).name,strII ));
        xT_missing = importdata(fullfile(pwd, listdir(i).name, strII_missing ));
        x1 = importdata(fullfile(pwd, listdir(i).name, strV));
        fprintf('loaded from: %s\n', listdir(i).name);
    end
end


% 30 secs corresponds to N = 3750.

% Plotting current data.
subplot(3,1,1);
plot(xT);
xlim([length(x2)-12500, length(x2)]);
subplot(3,1,2);
plot(x1);
xlim([length(x2)-12500, length(x2)]);
subplot(3,1,3);
plot(x2);
xlim([length(x2)-12500, length(x2)]);


N = 2;
%plot([xT, x1(1:size(xT)),x2(1:size(xT))])
bi = customRLS(xT, x1(1:size(xT)),x2(1:size(xT)), N);

% Reconstructing the last 30 secs
% 30 secs corresponds to N = 3750 samples.


M = 3750;
xhat = zeros(1,3751);
counter = 1;
for n = 67500:71250-1
   % xhat(n) = xhat(n) + sum(ai*x1prim(1:n));
    xhat(counter) = sum(bi'*x2(n-N:n-1));
    counter = counter +1;

%     for i = 1:n
%         xhat(n) = xhat(n) +  ai*x1prim(n-i);
%         xhat(n) = xhat(n) +  bi*x2prim(n-i);
%     end
end

subplot(3,1,1);
plot(x2(67500:71250));
subplot(3,1,2);
plot(xhat);
subplot(3,1,3);
plot(xT_missing);
%

