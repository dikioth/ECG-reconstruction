
     /* netdimsandfilenames.h - read thefile names where we will keep data for a record
        Copyright (C) 2010  Rui Rodrigues <rapr@fct.unl.pt>
        This software is released under the terms of the GNU General
        Public License (http://www.gnu.org/copyleft/gpl.html).
     */
struct netdimsandfilenames{

  int nsignals,patchsize,nhidden0,nhidden1,jump;

  string trainingdatafile,rbmvislinearhiddendatafile,netvislinearweights,netlogisticweights,lasterrorfile,
    netvislineardatafwdfile, autoencoderallcoeficients,autoencoderallcoeficientsgsl,
  patchdatafile,
  fwdautoencoderdatafile,backpropautoencodercoefficientsfile,
  fwdautoencoderdata_afterbackprop_file,
  autoencoderfirstlayerdatafile,autoencodersecondlayerdatafile,fwdautoencoderdata_after_rbmbackprop_file,
  patch_fwd_data_rbmvislinearfile, backprop_rbmvislinearweightsfile,
  backprop_rbmlogisticweightsfile,  backproprbmvislinearimagefile,
  backproprbmlogisticimagefile;
};

void read_datafile(ifstream&in,netdimsandfilenames& A);
//in netdimsandfilenames.cpp
