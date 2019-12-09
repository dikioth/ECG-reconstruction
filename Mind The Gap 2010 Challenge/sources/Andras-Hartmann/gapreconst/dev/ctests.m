function ctests(nr)
%function ctests(nr)
%Runs the estimation on dataset c.
%Without parameter it runs all the tests, with parameter it runs the
%given group of 10 signal, this is to optimize multi-threaded running.
%Note that both the matlab version and the original datasets are needed.
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

rand('twister',sum(100*clock));
randn('state',sum(100*clock));
if nargin == 0
    from = 1
    to = 100
else
    from = 1+(nr-1)*10
    to = from + 9
end
fid = fopen('set-c/RECORDS','r');
A = textscan(fid,'%s');
A = A{:};
fclose(fid);
files = cell2mat(A);
Qs = [];
for filenr = from:to

    st = files(filenr,:);

    %only uncomment this if the tests were run at least once and you want only upgrade results under specified score
    %if we have a good enough score, we only try to improve the bad ones
    %{
    if exist(['estimate' st '.mat'])
	load(['estimate' st '.mat']);
	if (q1_old + q2_old > 1.95)
	    continue;
	end;
    end;
    %}

    tic
    [ff q1 q2 q1_ q2_] = runctest(st,5000);
    toc
    Qs = [Qs;[q1 q2 q1_ q2_]]
end
