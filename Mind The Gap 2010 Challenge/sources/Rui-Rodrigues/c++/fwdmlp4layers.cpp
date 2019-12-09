
#include <stdio.h>
#include <string>

#include <sys/time.h>

using namespace std;

#include<cassert>
#include <iostream>
#include <fstream>
#include <vector>
#include "gradgsl.h"
#include "netdimsandfilenames.h"

void checkfstream(ofstream& file_io,const char* filename);
//in ../geral/checkfstream.cpp

void checkfstream(ifstream& file_io,const char* filename);
//in ../geral/checkfstream.cpp

void read_datafile(ifstream&in,netdimsandfilenames& A);
//in netdimsandfilenames.cpp


void readgslvectormatrix(FILE* readfromfile, gsl_vector* v,vector<int>&sizes);
//save a vector that contains several gslmatrices 
void savegslvectormatrix(FILE* writetofile, gsl_vector* v,vector<int>&sizes);
int givesize_gslvector_infile(FILE* readfromfile);
void readgslmatriz(const char* filename,gsl_matrix*&m);
void writegslmatriz(const char* filename,gsl_matrix*m);
void writegslvector(const char* filename,gsl_vector*m);
//in iosgslvectormatrix.cpp



void reconstructfrompatches_startingfrom(gsl_vector*longsignal, gsl_matrix*patches,
					 int start, int comp, int jump);
//in reconstructfrompatchesstartingfrom.cpp

// -------------------------------------------------------------------


void interpolatepatchdata(gsl_matrix*&patchdata,int newpatchsize,int step);
//in interpolate.cpp


int main(int argc, char ** argv){


  try{

if(argc<4){

    cout<<" must be called with argument signal1 signalaim folder name and optionaly with 'aim new patchsize'!"<<endl;

    exit(0);
  }

//in the case that the aim signal was subsample before training it must now be interpolate to recover original patch size
//before being reconstructed the long signal

  //time -----------------
  struct timeval start, end;
  gettimeofday(&start, NULL);
  //----------------------


 //get filenames to extract dimensions and else--


  //signal1

  string aa1=argv[1];
  string cc=argv[3];
  string b1=aa1.append(".txt");
  string d1="../";
  d1.append(cc);
  d1.append("/");
  d1.append(b1);

  cout<<"reading configuration data from "<<d1.c_str()<<endl;

  netdimsandfilenames A1;

  ifstream reading1(d1.c_str());

  read_datafile(reading1,A1);

  checkfstream(reading1,d1.c_str());

  reading1.close();
  

  //signal2


  string aa2=argv[2];
  string b2=aa2.append(".txt");
  string d2="../";
  d2.append(cc);
  d2.append("/");
  d2.append(b2);

  cout<<"reading configuration data from "<<d2.c_str()<<endl;

  netdimsandfilenames A2;

  ifstream reading2(d2.c_str());

  read_datafile(reading2,A2);

  checkfstream(reading2,d2.c_str());

  reading2.close();


  //-------------change in case of different number of layers---------------

  int nhidden0=A1.nhidden0, nhidden1=A1.nhidden1, 
    nhidden2=A2.nhidden1, nhidden3=A2.nhidden0, jump=A1.jump;

  int ninputs=A1.nsignals*A1.patchsize, noutputs=A2.patchsize;//only works for a single signal in the output 

  size_t numhidlayers=4;

  int dimensions[]={ninputs,nhidden0,nhidden1,nhidden2,nhidden3,noutputs};

  //--------------------------------------------------------------------


 //----------------------------------------------------
  //load inputdata

  gsl_matrix*inputdata;

  //shoul be this one
  string inputdatafile="_allpatches.txt";


  string base="../";
  base.append(cc);
  base.append("/");
  base.append(argv[1]);
  base.append(inputdatafile);

  cout<<"loading input data from "<<base.c_str()<<endl;

  readgslmatriz(base.c_str(),inputdata);

  //ifstream  indata(base.c_str());
  //load(indata,inputdata);
  //checkfstream(indata,base.c_str()); 
  //indata.close();

  if((int) inputdata->size2!=ninputs){
    cout<<"inputdata is not compatible with ninputs!"<<endl;
    exit(1);
  }


  //----------------------------------------------------------


  //----------load weights---------------------------

  //file for mlp4layerweights
  string f="../";
  f.append(cc);
  f.append("/mlp4layers_");
  string signal1=argv[1];
  string signal2=argv[2];
  f.append(signal1);
  f.append("_");
  f.append(signal2);
  f.append(".txt");
 
 



  size_t npatches=inputdata->size1;

  //size_t numbatches=1;

  //parametersgradgsl td(numhidlayers,dimensions,npatches);    

  //load weights 

  int numweights=(ninputs+1)*nhidden0+(nhidden0+1)*nhidden1+(nhidden1+1)*nhidden2+(nhidden2+1)*nhidden3+(nhidden3+1)*noutputs;


  gsl_vector * minimumweights=gsl_vector_alloc (numweights);


    vector<int> sizes;

    FILE * readweights = fopen(f.c_str(), "r");

    cout<<"loading weights from "<<f.c_str()<<endl;

    readgslvectormatrix(readweights, minimumweights,sizes);

  //debug
  //cout<<"minimuweights norm is "<<gsl_blas_dnrm2 (minimumweights)<<endl;
 

  //fwd data

  parametersfwdgsl tf(numhidlayers,dimensions,npatches); 


    
  gsl_matrix_memcpy(&tf.reallayerdata[0].matrix,inputdata);

  fwdgsl_vislinear(minimumweights,&tf,tf.fwd_data);

  gsl_vector_free(minimumweights);  
  
 

  //file for fwd_data
  string fdata="../";
  fdata.append(cc);
  fdata.append("/fwd_allpatches_mlp4layers_");
  fdata.append(signal1);
  fdata.append("_");
  fdata.append(signal2);
  fdata.append(".txt");

  cout<<"writing fwd allpatches in file"<<fdata.c_str()<<endl;

  writegslmatriz(fdata.c_str(),tf.fwd_data);




  //interpolate patches data when necessary 

  int newpatchsize;


  int comp;


  if(argc>4){

    newpatchsize=atoi(argv[4]);

    int step=(newpatchsize-1)/(noutputs-1);



    if(newpatchsize>noutputs){

      interpolatepatchdata(tf.fwd_data,newpatchsize,step);

      comp=(npatches-1)*jump+newpatchsize;
    }
    else
    comp=(npatches-1)*jump+noutputs;

  }
  else
    comp=(npatches-1)*jump+noutputs;
  


  gsl_vector*recsignal=gsl_vector_alloc(comp); 



  //start allways at zero
  reconstructfrompatches_startingfrom(recsignal, tf.fwd_data,0, comp, jump);


  //file for reconstructed long signal
  string recsg="../";
  recsg.append(cc);
  recsg.append("/reconstructed_long_signal_");
  recsg.append(signal1);
  recsg.append("_");
  recsg.append(signal2);
  recsg.append(".txt");

  cout<<"writing reconstructed signal in "<<recsg.c_str()<<endl;

  gsl_matrix_view aauuxx=gsl_matrix_view_vector(recsignal,1,recsignal->size);
  
  writegslmatriz(recsg.c_str(),&aauuxx.matrix); 



  gsl_vector_free(recsignal);

  //time----------------------

  gettimeofday(&end, NULL);

  int seconds  = end.tv_sec  - start.tv_sec;
  int useconds = end.tv_usec - start.tv_usec;

  int mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;

  cout<<"Elapsed time: "<<mtime<<" milliseconds\n"<<endl;

  //--------------------------



  }
  catch(int i){

    cout<<"caught "<<i<<endl;
  }

 
  return 0;
}


