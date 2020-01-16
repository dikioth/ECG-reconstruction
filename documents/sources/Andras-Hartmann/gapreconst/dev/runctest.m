function [ff q1 q2 q1_ q2_] = runctest(st, slength)
%function [ff q1 q2 q1_ q2_] = runctest(st, slength)
%Runs one estimation on the file st, the length of fitting is slength-3750
% output
% ff: estimate for the gap
% q1: actual predicted q1 score
% q2: actual predicted q2 score
% q1_: previously predicted q1 score
% q2_: previously predicted q2 score
% the function also saves the estimate in a file and other information about the result
%
%Copyright (C) 2010  Andras Hartmann <hdbandi@gmail.com>
%
%This file is part of GapReconst, a software for
%reconstructing short loss of physiological signals.
%
%GapReconst is free software: you can redistribute it and/or modify
%it under the terms of the GNU General Public License as published by
%the Free Software Foundation, either version 3 of the License.
%
%This program is distributed in the hope that it will be useful,
%but WITHOUT ANY WARRANTY; without even the implied warranty of
%MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%GNU General Public License for more details.
%
%You should have received a copy of the GNU General Public License
%along with this program.  If not, see <http://www.gnu.org/licenses/>.

load(['challenge/2010/set-c/' st 'm.mat']);

%remove signals without information to reduce computational complexity
toremove = [];
for i = 1:size(val,1)
	if val(i,end-7500:end-3750) == zeros(1,3751)
	    toremove = [toremove i]
	end;
end;

if (length(toremove)>0)
    val(toremove,:) = [];
end;

%find what is missing
for i = 1:size(val,1)
	if val(i,end-3749:end) == zeros(1,3750)
		break;
	end;
end;
ind = i;

% These are the inputs to the filter network
%x	: the history signal
%xx	: the concurrent signal to the gap
%xxx	: 30s of the signal to compare results fitted on different length
x = val(:,end-slength:end-3750);
xx = val(:,end-3749:end);
xxx = val(:,end-7500:end-3750);
x(ind,:) = [];
xx(ind,:) = [];
xxx(ind,:) = [];

%remove mean, so we do not have to identify this
mx = mean(x,2);
mxx = mean(xx,2);
mxxx = mean(xxx,2);
x = x - repmat(mx,1,size(x,2));
xx = xx - repmat(mxx ,1,size(xx,2));
xxx = xxx - repmat(mxxx ,1,size(xxx,2));

% These are the desired outputs used for fitting
%y	: history of the gap
%yyy	: 30s of the history to compare results fitted on different length
y = val(ind,end-slength:end-3750);
yyy = val(ind,end-7500:end-3750);

my = mean(y);
sy = std(y);

%check if the function is not identical
if (sy == 0)
    ff = ones(1,3750)*my;
    q1 =1; q2=1; q1_old=1; q2_old=1;
    save(['estimate' st], 'q1_old', 'q2_old');
    saveascii(ff',['reconst' st], 4);
    return;
end

% remove mean from y
y = y-my;

%fitting x to y to find the coefficients
[f,b,a] = genalg(x, y);

%ff	: reconstruction of the gap
%fff	: reconstruction of 30s of the history
ff = multifilter(b, a, xx);
fff = multifilter(b, a, xxx);

%add the mean again
ff = ff+my;
fff = fff+my;

%estimate the error on the last 30s of the history
ccf = corrcoef(yyy,fff);
q1 = 1-mse(yyy-fff)/var(yyy)
q2 = ccf(2)

q1_ = 0;
q2_ = 0;

%check if the current reconstruction is better than the previous
if exist(['estimate' st '.mat'])
    load(['estimate' st '.mat']);

    q1_ = q1_old
    q2_ = q2_old
    %if the old estimates are better, we don't do anything
    if max(q1,0)+max(q2,0) <= max(q1_old,0)+max(q2_old,0)
	return;
    end
end

q1_old = q1;
q2_old = q2;

%saving the data
save(['estimate' st], 'q1_old', 'q2_old');
save(['history' st], 'fff');

saveascii(ff',['reconst' st], 4);
