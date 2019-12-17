function p = getpatient(pn)
% loading data (sampled at Fs = 125Hz).

% Files are:
%       Variable	Description             File Name
%   =========================================================
%       xT        Target signal.          ECG_X_AVR.mat
%       x1        Reference signal 1.     ECG_X_II.mat
%       x2        Reference signal 2.     ECG_X_V.mat
%       xTm       Missing part.           ECG_X_II_missing.mat
%   ----------------------------------------------------------
%
%   where X corresponds to patient number and found in folder 'ECG_X_aYZ'
%   Folders are: ECG_ ...
%         [1_a02 | 2_a03 | 3_a06  | 4_a07 | 5_a08 |6_a09 | 7_a11 |8_a12]


% Get strings
strAVR = ['ECG_', num2str(pn), '_AVR.mat'];
strII = ['ECG_', num2str(pn), '_II.mat'];
strII_m = ['ECG_', num2str(pn), '_II_missing.mat'];
strV = ['ECG_', num2str(pn), '_V.mat'];

% Folder is in 'data' folder.
listdir = dir('data');
for i = 1:length(listdir)
    if strfind(listdir(i).name, ['ECG_', num2str(pn)]) == 1
        p.x2 = importdata(fullfile(pwd, 'data', listdir(i).name, strAVR));
        p.xT = importdata(fullfile(pwd, 'data', listdir(i).name, strII));
        p.xTm = importdata(fullfile(pwd, 'data', listdir(i).name, strII_m));
        p.x1 = importdata(fullfile(pwd, 'data', listdir(i).name, strV));
    end
end

% Index of the last 30 secs
p.indexMissingPart = length(p.x1)-length(p.xTm):length(p.x1);

end