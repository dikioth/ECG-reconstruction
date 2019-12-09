function rnum = unidrnd(max)
%function rnum = unidrnd(max)
%Random numbers from the discrete uniform distribution
%0< rnum <= max
%NOTE:
%A function with identical name and similar functionality
%can be found in Statistics Toolbox.
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
rnum = 0;

while rnum == 0
    rnum = ceil(rand * max);
end

