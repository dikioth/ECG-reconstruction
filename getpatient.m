function [xT, x1,x2,xTmissing] =  getpatient(p)

% Get strings
strAVR = ['ECG_', num2str(p),'_AVR.mat'];
strII = ['ECG_', num2str(p), '_II.mat'];
strII_missing =  ['ECG_', num2str(p), '_II_missing.mat'];
strV =  ['ECG_', num2str(p), '_V.mat'];

% Folder is same folder as main.m
listdir = dir;
for i = 1:length(listdir)
    if strfind(listdir(i).name, ['ECG_', num2str(p)]) == 1
        x2 = importdata(fullfile(pwd, listdir(i).name, strAVR ));
        xT = importdata(fullfile(pwd, listdir(i).name,strII ));
        xTmissing = importdata(fullfile(pwd, listdir(i).name, strII_missing ));
        x1 = importdata(fullfile(pwd, listdir(i).name, strV));
    end
end

end