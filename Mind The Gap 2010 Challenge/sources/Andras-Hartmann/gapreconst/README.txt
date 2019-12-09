DESCRIPTION:

 This program is called GapReconst, a software written in MATLAB
 for reconstructing short loss of physiological signals
 using the Physionet Mind The Gap Challenge 2010 c dataset.

 MATLAB	:  Version 7.6.0.324 (R2008a)
 OS	:  GNU/Linux
 AUTHOR	:  Andras Hartmann
 CONTACT:  hdbandi@gmail.com


LICENSE:

 GapReconst is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.


DEPENDENCIES:

 - Signal Processing toolbox

 - saveascii, an open source program that saves/displays a matrix array
   written by Carlos Adrian Vargas Aguilera.
   This distribution contains a version of this software.

 - Internet connection to download the data.

 - It is strongly recommended to have rsync and the WFDB package from
   PhysioNet installed to download the dataset in appropriate format.

INSTALLING:

 Simply unzip the software in a directory,
 this directory is referred later as software root.


RUNNING:

 LINUX:
  Linux scripts are provided to download physionet data,
  and do the reconstruction of missing signals on the
  dataset c of Mind The Gap Physionet Challenge 2010

  getdata.sh	- downloads dataset from physionet

  getmcdata.sh	- downloads dataset c in matlab format

  runc.slurm	- batch file to run tests on dataset c.
  You can run it as standalone script or add it to slurm,
  see manpage of slurm for details. Note that the location
  of your MATLAB distribution has to be set

  runmec.slurm	- batch file to run tests multithreaded
  on dataset using slurm, see manpage of slurm for details.

  To use these scripts it is suggested to copy them in a directory, eg. data.
  To download the dataset in MATLAB format use the method of WFDB
  (WFDB from physionet should be also installed).

  Your directory structure will be somewhat like

  -software root
  |
  x-- dev	: .m files for reconstruction of the gaps
  |
  x-- script	: scripts for running on linux
  |
  x-- data	: folder for the reconstruction
  |
  x-- set-c	: set-c in binary format
  |
  x-- challenge	: directory structure containing set-c in MATLAB format

  You can use the following commands from the software root:

  $mkdir data
  $cp script/* data/
  $cd data
  $./getdata.sh
  $./getmcdata.sh
  $./runc.slurm

  Or you can simply run runme.sh after adding the path of your matlab distribution
  to scripts/runc.slurm.

  If you did everything correctly, the script will start with downloading the data
  and run the reconstruction.
  Then sit back and relax, because the reconstruction algorithm can take a while:
  on an Intel(R) Xeon(R) E5310  @ 1.60GHz CPU the reconstruction of one signal takes
  around 12 minutes, thus the reconstruction of all 100 signals is about 20 hours.
  The reconstructed time-series are then put to data folder as reconstcxx files where
  xx notes for the number of the input signal.


 OTHER OPERATING SYSTEM:

  Note that however these scripts are for linux, you are able to run the code on
  another operating systems as well. To do this you have to do the following:

  - download the challenge dataset (see the directory structure above)

  - start MATLAB

  - add the location of the dev and saveascii directories to your MATLAB path:
    >> addpath ../dev/; addpath ../dev/saveascii/;

  - run ctests without parameter (eg. in one thread only):
    >> ctests;
