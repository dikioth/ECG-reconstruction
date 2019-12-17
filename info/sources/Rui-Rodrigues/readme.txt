                                       [Prepare software]

There are C++  and octave/matlab programs that must be used.

To compile C++ programs you need first to install GSL (gnu scientific library) and acordingly you need to configure one macro in the provided 'makefile' that compiles all the C++ programs. You also need to use 'bash' to run some scripts (there exist freely available versions for MSWINDOWS (included in Cygwin for example)

Octave/matlab programs: if you use octave you need to change the extension of the programs 'makefilnames.octave',
'makenetdimsandfilenames.octave' and 'viewresults.octave' respectively to 'makefilnames.m',
'makenetdimsandfilenames.m' and 'viewresults.m' and the same you have to do if you use matlab with the files
'makefilnames.matlab', 'makenetdimsandfilenames.matlab' and 'viewresults.matlab'. This is due to an incompatibility in the way matlab andd octave deal with '\n'.


                                      [Run the software]

To fill in  the gap in the record c12 (an example) you need to follow the next steps.


[1] Produce the data (for the training step)

1.1 You need to download from 'physionet atm' into the directory c12 the files c12m.mat and c12m.info .  
If instead you want to run a record from set A, in the '.info' file you need to change the names from ecg signal from 'ECG II' to 'II'. 


The dirctory structure should be:  basedir/c12, basedir/octave_matlab, basedir/c++


1.2 From octave/matlab you must run in the directory octave_matlab the folowing commands:

      load ../c12/c12m.mat

      producedata(val,{'ICP','PLETH'},'RESP','c12');

 The last instruction assumes we'll reconstruct target signal (RESP) using ICP and PLETH. The file 
'used_signals_for_reconstruction.txt' gives for each record the list of signals used to reconstruct the target signal




[2] Training step

-In the c++ directory you must run the following 'bash script' with the given arguments 

 ./trainforresp.sh ICP PLETH c12

 (if target signal were not RESP you would run  script 'trainforallbutresp.sh' instead of trainforresp.sh)

(wait many many hours or some days depending on the computer, if the target were not 'RESP' it would be much faster)


[3] producing the reconstructed signal:


3.1 In the c++ directory run 

 ./fwdmlp4layers otherICPPLETH aim c12 375

(if target signal were not RESP the last argument would be 125 instead of 375)


3.2 In the octave_matlab directory run in octave/matlab the following commands

   load ../c12/c12m.mat

   [patchdata, fwdallpatches, reclongsignal]=viewresults('c12',val,'otherICPPLETH',1)
  
  
 (the last argument is 1 because RESP is the first signal in record c12 otherwise you would change accordingly to its order in the record)


   Now your reconstructed signal should be in the file basedir/c12/missingsignal_otherICPPLETH_aim.txt

   In the case of 'RESP' sometimes the reconstruction signal values are not between the minimum and maximum value and in that case we should run the file 'saturate' (octave or matlab version, after renaming it). 

 
3.3 If you want to see how the neuralnetwork learned to reconstruct the target signal (RESP) from

the signal ICP and PLETH, continue giving the following commands in octave/matlab


    x=1:1250;

    w=31250;plot(x,val(1,x+w),'k',reclongsignal(x+w),'r')

and you'll see a patch of original signal RESP in black and the learned reconstructed signal in red. To see a different patch just change the value of 'w'.  


If the target signal were not RESP you need to add another command and so it should be


    x=1:1250;

    reclongsignal=[zeros(1,125),reclongsignal];

    w=31250;plot(x,val(1,x+w),'k',reclongsignal(x+w),'r')



