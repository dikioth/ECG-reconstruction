
     /* netdimsandfilenames.cpp - read thefile names where we will keep data for a record
        Copyright (C) 2010  Rui Rodrigues <rapr@fct.unl.pt>
        This software is released under the terms of the GNU General
        Public License (http://www.gnu.org/copyleft/gpl.html).
     */


#include <fstream>
#include <iostream>
using namespace std;

#include "netdimsandfilenames.h"

void checkfstream(ofstream& file_io,const char* filename);
//in ../geral/checkfstream.cpp

void checkfstream(ifstream& file_io,const char* filename);
//in ../geral/checkfstream.cpp


void read_datafile(ifstream&in,netdimsandfilenames& A){

  const char file[]="netdims and file names file";


  in>>A.nsignals>>A.patchsize>>A.nhidden0>>A.nhidden1>>A.jump;

  in>>A.trainingdatafile>>A.rbmvislinearhiddendatafile>>A.netvislinearweights;
  in>>A.netlogisticweights>>A.lasterrorfile;
  in>>A.netvislineardatafwdfile>>A.autoencoderallcoeficients>>A.autoencoderallcoeficientsgsl;
  in>>A.patchdatafile;
  in>>A.fwdautoencoderdatafile>>A.backpropautoencodercoefficientsfile;
  in>>A.fwdautoencoderdata_afterbackprop_file>>A.fwdautoencoderdata_after_rbmbackprop_file;
  in>>A.autoencoderfirstlayerdatafile>>A.autoencodersecondlayerdatafile;
  in>>A.patch_fwd_data_rbmvislinearfile;
  in>>A.backprop_rbmvislinearweightsfile>>A.backprop_rbmlogisticweightsfile;
  in>>A.backproprbmvislinearimagefile;
  in>>A.backproprbmlogisticimagefile;

  checkfstream(in,file);

  //debug
  //cout<<"configuration file ok!"<<endl;
  
}
