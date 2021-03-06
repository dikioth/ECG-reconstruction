function p = getpatient(pn)
% getpatient: Loads patient data (sampled at Fs = 125Hz).
%   Use:    p = getpatient(pn)
%
%   Input:  pn: scalar value. The number of patient to be loaded.
%                allowed are: 1, 2, 3, 4, 5, 6, 7, 8.
%
%   Output: p: struct variable. The data of the chose patient containing
%                - target signal with mean substracted: xTzm. 
%                - The mean of the target signal: xTmean.
%                - first reference signal w/ mean substracted: x1zm
%                - Second reference signal w/ mean zusbracted: x2zm 
%                - The mean of the target signal.
%
%   Files loaded are stored in the folder 'data'. Files are:
%
%       Variable    Description             File Name
%   ===========================================================
%       xT          Target signal.          ECG_X_AVR.mat
%       x1          Reference signal 1.     ECG_X_II.mat
%       x2          Reference signal 2.     ECG_X_V.mat
%       xTm         Missing part.           ECG_X_II_missing.mat
%   ------------------------------------------------------------
%
%   where X corresponds to patient number and files are 
%   found in folder 'ECG_X_aYZ'
%   Folders are: ECG_ ...
%         [1_a02 | 2_a03 | 3_a06  | 4_a07 | 5_a08 |6_a09 | 7_a11 |8_a12]


% Get strings
strAVR = ['ECG_', num2str(pn), '_AVR.mat'];
strII = ['ECG_', num2str(pn), '_II.mat'];
strII_m = ['ECG_', num2str(pn), '_II_missing.mat'];
strV = ['ECG_', num2str(pn), '_V.mat'];

% Assuming folders are in 'data' folder.
listdir = dir('data');
for i = 1:length(listdir)
    if strfind(listdir(i).name, ['ECG_', num2str(pn)]) == 1
        p.x2 = importdata(fullfile(pwd, 'data', listdir(i).name, strAVR));
        p.xT = importdata(fullfile(pwd, 'data', listdir(i).name, strII));
        p.xTm = importdata(fullfile(pwd, 'data', listdir(i).name, strII_m));
        p.x1 = importdata(fullfile(pwd, 'data', listdir(i).name, strV));
    end
end

p.xTzm = p.xT - mean(p.xT); % Target signal w/ mean substracted.
p.x1zm = p.x1 - mean(p.x1); % Reference 1 siganl w/ mean substracted.
p.x2zm = p.x2 - mean(p.x2); % Reference 2 signal w/ mean substracted.
p.xTmean = mean(p.xT);      % mean of target signal.
end