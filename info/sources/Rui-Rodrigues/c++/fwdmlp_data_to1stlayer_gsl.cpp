
     /* fwdmlp_data_to1stlayer_gsl.cpp - forward input data to the first
	hidden layer of an autoencoder
        Copyright (C) 2010  Rui Rodrigues <rapr@fct.unl.pt>
        This software is released under the terms of the GNU General
        Public License (http://www.gnu.org/copyleft/gpl.html).
     */




#include <iostream>
#include "../geral/matrix.h"
#include <string.h>
#include <sys/time.h>

#include <gsl/gsl_blas.h>
#include "gradgsl.h"


using namespace std;

#include "netdimsandfilenames.h"

//--------------------------------------------



void checkfstream(ofstream& file_io,const char* filename);
//in checkfstream.cpp

void checkfstream(ifstream& file_io,const char* filename);
//in checkfstream.cpp

void loadfirstmatrixfromfile(const char * filename, gsl_vector* weights,int*nrows, int * ncols);
//down in this file

void readgslmatriz(const char* filename,gsl_matrix*&m);
//in iogslvectormatrix.cpp

void writegslmatriz(const char* filename,gsl_matrix*m);
//in iosgslvectormatrix.cpp


int main(int argc, char ** argv){


  try{

if(argc<3){

    cout<<" must be called with argument signalname and folder name !"<<endl;

    exit(0);
  }


 //get filename to extract dimensions and else--

  string aa=argv[1];
  string cc=argv[2];
  string b=aa.append(".txt");
  string d="../";
  d.append(cc);
  d.append("/");
  d.append(b);

 cout<<"reading configuration data from "<<d.c_str()<<endl;

  netdimsandfilenames A;

  ifstream reading(d.c_str());

  read_datafile(reading,A);

  checkfstream(reading,d.c_str());

  reading.close();


 int nsignals=A.nsignals, patchsize=A.patchsize, nhidden0=A.nhidden0;
 //, nhidden1=A.nhidden1;

  int ninputs=nsignals*patchsize; 
 
  int nrows[]={ninputs+1}; 
  //int nrows[]={ninputs+1,nhidden0+1};


  int ncols[]={nhidden0};
  //int ncols[]={nhidden0,nhidden1};


  //load training data

  gsl_matrix * data;

  readgslmatriz(A.patchdatafile.c_str(),data);
 

  //check
  if((int) data->size2!=ninputs){
    cout<<"training data is not compatible with ninputs!"<<endl; 
    exit(1);
  }




  int npatches=data->size1;
  
 


  //load weights---------------------------------------------------

 
  int numcoefficients=nrows[0]*ncols[0];
  //int numcoefficients=nrows[0]*ncols[0]+nrows[1]*ncols[1];

  gsl_vector *weights=gsl_vector_alloc(numcoefficients);


  loadfirstmatrixfromfile(A.backpropautoencodercoefficientsfile.c_str(), weights,nrows, ncols);


  
  //-----

  //fwd data

  int dimensions[]={ninputs,A.nhidden0};

  parametersfwdgsl tf(0,dimensions,npatches); 

  gsl_matrix_memcpy(&(tf.reallayerdata[0].matrix),data);


		     
  fwdgsl_vislogistic(weights,&tf,tf.fwd_data);


  cout<<"saving fwd data on file:  "<<A.autoencoderfirstlayerdatafile.c_str()<<endl;


  writegslmatriz(A.autoencoderfirstlayerdatafile.c_str(),tf.fwd_data);

  gsl_vector_free(weights);  
  gsl_matrix_free(data); 
  }
 catch(int i){

    cout<<"caught "<<i<<endl;
  }

  return 0;

}




void loadfirstmatrixfromfile(const char * filename, gsl_vector* weights,int*nrows, int * ncols){


  FILE * readweights = fopen(filename,"r");

  int num_matrices;

  fscanf(readweights,"%u",&num_matrices);

  vector<int> sizes;

  sizes.resize(1+2*num_matrices);

  for(int i=1;i<1+2*num_matrices;++i)
    fscanf(readweights,"%d",&sizes[i]);


  //check dimensions
  bool correct=true;

  if(num_matrices!=4) correct=false;

  int numcoefficients=nrows[0]*ncols[0];


  if(sizes[1]!=nrows[0]) 
    correct=false;
  if(sizes[2]!=ncols[0]) 
    correct=false;
  

  if(numcoefficients!= (int) weights->size)
    correct=false;
  
  if(!correct){
    cout<<"dimension of matrices dimension loaded from file are not correct!"<<endl;
    char a; cin>>a;
  }

  int failure=gsl_vector_fscanf(readweights, weights);

  if(failure){    
    cout<<"couldn t read gsl_vector!"<<endl; 
    exit(1);
  }
  
}
 
