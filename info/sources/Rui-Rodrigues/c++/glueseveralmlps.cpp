
     /* glueseveralmlps.cpp - create a new mlp putting side by side several mlps
        Copyright (C) 2010  Rui Rodrigues <rapr@fct.unl.pt>
        This software is released under the terms of the GNU General
        Public License (http://www.gnu.org/copyleft/gpl.html).
     */

#include <string>
#include <iostream>
#include <fstream>

using namespace std;

#include "netdimsandfilenames.h"
#include "mlp.h"


void checkfstream(ofstream& file_io,const char* filename);
//in checkfstream.cpp

void checkfstream(ifstream& file_io,const char* filename);
//in checkfstream.cpp


void read_datafile(ifstream&in,netdimsandfilenames& A);
//in netdimsandfilenames.cpp


mlp  glue2mlps(mlp&mlp1,mlp&mlp2);
//in glue2mlps.cpp


int main(int argc, char ** argv){

 cout<<"arguments must be component  signals names, new signal name and finally folder name"<<endl;

 unsigned k=argc;


 //k-3 is the number of mlps we want to glue


 netdimsandfilenames A[2];


  string folder=argv[k-1];

  string d="../";
  d.append(folder);
  d.append("/");



  //first mlp
  string b=d;
  string aa=argv[1];
  b.append(aa);
  b.append(".txt");

  cout<<"reading configuration data from "<<b.c_str()<<endl;

  ifstream reading0(b.c_str());
  read_datafile(reading0,A[0]);
  checkfstream(reading0,b.c_str());
  reading0.close();

  unsigned nsignals=A[0].nsignals;
  if(nsignals!=1){

    cout<<"this function doesn t work when nsignals>1"<<endl;
    exit(-1);
  }

  unsigned patchsize=A[0].patchsize,
    nhidden0=A[0].nhidden0, nhidden1=A[0].nhidden1;



  unsigned layersdimensions[]={patchsize,nhidden0,nhidden1,nhidden0,patchsize};

  vector<unsigned> dimlayers(layersdimensions,layersdimensions+5);

  mlp net0(3,dimlayers);


  net0.loadtxtweights(A[0].backpropautoencodercoefficientsfile.c_str());


  for(unsigned i=1;i<k-3;++i){
  
   //get filename to extract dimension and else--

    aa=argv[i+1];

    b=d;
    b.append(aa);
    b.append(".txt");

    cout<<"reading configuration data from "<<b.c_str()<<endl;

    ifstream reading1(b.c_str());

    read_datafile(reading1,A[1]);

    checkfstream(reading1,b.c_str());

    reading1.close();  

    unsigned nsignals1=A[1].nsignals;
    if(nsignals1!=1){
      cout<<"this function doesn t work when nsignals>1"<<endl;
      exit(-1);
    }


    unsigned patchsize1=A[1].patchsize,
      nhidden01=A[1].nhidden0, nhidden11=A[1].nhidden1;


    
    unsigned layersdimensions1[]={patchsize1,nhidden01,nhidden11,nhidden01,patchsize1};

    vector<unsigned> dimlayers1(layersdimensions1,layersdimensions1+5);

    mlp net1(3,dimlayers1);


    net1.loadtxtweights(A[1].backpropautoencodercoefficientsfile.c_str());

    
    net0=glue2mlps(net0,net1);

  }


  //put weights of new mlp in disk

  b=d;
  aa=argv[k-2];
  b.append(aa);
  b.append(".txt");

  netdimsandfilenames B;

  ifstream reading(b.c_str());
  read_datafile(reading,B);
  checkfstream(reading,b.c_str());
  reading.close();


  cout<<"Saving weights on file "<<B.backpropautoencodercoefficientsfile.c_str()<<endl;
  
  net0.savetxtweights(B.backpropautoencodercoefficientsfile.c_str());

  return 0;

}

