function [f,b,a] = genalg(x, y)
%[f,b,a] = genalg(x, y)
%genetic algorithm for determining the filter coefficients in the filtering network between x and y
% Outputs:
% f: estimate for y
% b, a: filter coefficients (see filter)
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

bdim = 4;
adim = 4;
zdim = size(x,1);
population = 100;

%Choosing the first population from 100*population to have an appropriate prior
bs = randn(100*population,bdim,zdim);
as = randn(100*population,adim,zdim);

%The actual iterations of the genetic algorithm
for iter = 1:2000;
	c = [];
st1 = size(as,1);
for i = 1:st1

        f = multifilter(squeeze(bs(i,:,:)),squeeze(as(i,:,:)),x);

	ccf = corrcoef(y,f);
	c(i) = mse(y-f)*(1-ccf(2,1));
end

% Setting up the next generation
newas = [];
newbs = [];

%population/5: survival of the fittest
pop5 = population/5;
for i = 1:pop5
	[value,index] = min(c);
%	uncomment this if you would like to have some info printed about the generations
%	if i == 1
%		value
%		index
%	end
	c(index) = [];
	if ~isnan(value)
		newas = [newas; as(index,:,:)];
		newbs = [newbs; bs(index,:,:)];
	end
end

%The rest of the population is set up as recombination of two random specimen
st1 = size(newas,1);
for i = st1+1:population
	mo = unidrnd(st1);%mother
	fa = unidrnd(st1);%father
	
		newas(i,:,:) = mean([newas(mo,:,:); newas(fa,:,:)]);
		newbs(i,:,:) = mean([newbs(mo,:,:); newbs(fa,:,:)]);

end

%Adding random (Gauss) mutation in three different way
%some mutation to all all coeffitients
newas(pop5+1:2*pop5,:,:) = newas(pop5+1:2*pop5,:,:)+randn(pop5,adim,zdim)*0.001;
newbs(pop5+1:2*pop5,:,:) = newbs(pop5+1:2*pop5,:,:)+randn(pop5,bdim,zdim)*0.001;

%some more mutation to all coeffitients, this time with larger amplitude
newas(2*pop5+1:3*pop5,:,:) = newas(2*pop5+1:3*pop5,:,:)+randn(pop5,adim,zdim)*0.01;
newbs(2*pop5+1:3*pop5,:,:) = newbs(2*pop5+1:3*pop5,:,:)+randn(pop5,bdim,zdim)*0.01;

%some more mutation, this time with even larger amplitude, but adding only with 0.1 probability to the coeffitients
R1 = rand(pop5,adim,zdim);
R = zeros(pop5,adim,zdim);
R (find(R1>0.9)) = 1;
newas(3*pop5+1:4*pop5,:,:) = newas(3*pop5+1:4*pop5,:,:)+R.*randn(pop5,adim,zdim)*0.1;
R1 = rand(pop5,bdim,zdim);
R = zeros(pop5,bdim,zdim);
R (find(R1>0.9)) = 1;
newbs(3*pop5+1:4*pop5,:,:) = newbs(3*pop5+1:4*pop5,:,:)+R.*randn(pop5,bdim,zdim)*0.1;

%Replacing the actual generation with the new one
as = newas;
bs = newbs;

%This would be the condition to stop, but not used
%setting limit
%if (c(1) < 0.5)
%    break;
%end

end;
% the best filters to return
a = squeeze(as(1,:,:));
b = squeeze(bs(1,:,:));
