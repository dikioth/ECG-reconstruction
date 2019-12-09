     /* fwdhiddenlayerrbmvislinear.cpp - forward input data to the  hidden 
	layer of an rbm with linear visivle units 
        Copyright (C) 2010  Rui Rodrigues <rapr@fct.unl.pt>
        This software is released under the terms of the GNU General
        Public License (http://www.gnu.org/copyleft/gpl.html).
     */


#include <stdio.h>
#include <string>
#include <sys/time.h>
#include <iostream>
#include <fstream>

using namespace std;

#include "netrbm.h"
#include "netdimsandfilenames.h"


#ifdef _OPENMP
#include <omp.h>
#endif



void checkfstream(ofstream& file_io,const char* filename);
//in checkfstream.cpp

void checkfstream(ifstream& file_io,const char* filename);
//in checkfstream.cpp


void read_datafile(ifstream&in,netdimsandfilenames& A);
//in netdimsandfilenames.cpp


void logistic (gsl_matrix * m,size_t nrows,size_t ncols);
//down in this file

void writegslmatriz(const char* filename,gsl_matrix*m);
void readgslmatriz(const char* filename,gsl_matrix*&m);
void readgslvector(const char* filename,gsl_vector*m);
void writegslvector(const char* filename,gsl_vector*m);
//in iogslvectormatrix.cpp



// ----------------------------------------------------------------------------------------
//-----------------------CONFIGURE-----------------------------------------------------

const size_t batchsize=500;

const size_t numepochs=25;

const double epsilonweights=0.025;

const double epsilonbias=0.025;

const double momentum=0.2;

const double weightscost=0.0002;


//-----------------------------------------------------------------------------------







void fwdhiddenlayervislinear(netrbm&net,gsl_matrix*data);
//down in this file


void writegslmatriz(const char* filename,gsl_matrix*m);
//in iosvectormatrix.cpp




int main(int argc, char ** argv){


  try{

if(argc<3){

    cout<<" must be called with argument signal1 and aftert folder name !"<<endl;

    exit(0);
  }



  //time -----------------
  struct timeval start, end;
  gettimeofday(&start, NULL);
  //----------------------



 //get filename to extract dimensions and else--


  //signal

  string signal=argv[1];
  string cc=argv[2];
  string b1=signal;
  b1.append(".txt");
  string folder="../";
  folder.append(cc);
  folder.append("/");
  string d1=folder;
  d1.append(b1);

 
  cout<<"reading configuration data from "<<d1.c_str()<<endl;

  netdimsandfilenames A1;

  ifstream reading1(d1.c_str());

  read_datafile(reading1,A1);

  checkfstream(reading1,d1.c_str());

  reading1.close();
  

  unsigned ninputs=A1.nsignals*A1.patchsize;

  unsigned nhidden=A1.nhidden0;


  //load training data (patchdatafile);

  gsl_matrix * data;

  string allpatches=folder;
  allpatches.append(signal);
  allpatches.append("_allpatches.txt");


  string aim="aim"; 


  if(aim.compare(argv[1])==0)

    readgslmatriz(A1.patchdatafile.c_str(),data);

  else

    readgslmatriz(allpatches.c_str(),data);


  if(data->size2!=ninputs){
    cout<<"inputdata is not compatible with ninputs!"<<endl;
    exit(1);
  }


  size_t npatches=data->size1;




  //load weights (first matrix with pure weights then visible bias finally 
  //hidden bias 

  gsl_vector * vectorweightsandbias=gsl_vector_calloc (ninputs*nhidden+ninputs+nhidden);

  readgslvector(A1.netvislinearweights.c_str(),vectorweightsandbias);



#ifdef _OPENMP
  int maxnumthreads = omp_get_max_threads();
#else /* _OPENMP */
  int maxnumthreads = 4;
#endif /* _OPENMP */



 const unsigned blocksize=npatches/maxnumthreads;


 netrbm net(ninputs,nhidden, npatches,maxnumthreads, blocksize, npatches, vectorweightsandbias);



 

  fwdhiddenlayervislinear(net,data);



  


  writegslmatriz(A1.rbmvislinearhiddendatafile.c_str(),net.posprobs);


 
  gettimeofday(&end, NULL);

  int seconds  = end.tv_sec  - start.tv_sec;
  int useconds = end.tv_usec - start.tv_usec;

  int mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;

  cout<<"Elapsed time: "<<mtime<<" milliseconds\n"<<endl;




  //free

  gsl_matrix_free(data);

  gsl_vector_free(vectorweightsandbias);
  }

  catch(int i){
    cout<<"exception "<<i<<endl;
    exit(1);
  }


  return 0;

  }









  //-----------------------------------------------------------------------------------



void logistic (gsl_matrix * m){

  unsigned nrows=m->size1, ncols=m->size2;

  for(unsigned i=0;i<nrows;++i)
    for(unsigned j=0;j<ncols;++j)
      gsl_matrix_set(m,i,j,1/(1+exp((-1)*gsl_matrix_get(m,i,j))));

}




void fwdhiddenlayervislinear(netrbm&net,gsl_matrix*data){


  gsl_matrix_memcpy(net.permutedtdata,data);

  //initialize data

     net.batchtdata=gsl_matrix_submatrix(net.permutedtdata,0,0,net.npatches,net.ninputs);

     for(unsigned thread=0;thread<net.maxnumthreads;++thread)
       
       net.blocktdata[thread]= gsl_matrix_submatrix(&net.batchtdata.matrix,thread*net.blocksize,0,
						 net.sizesblocks[thread],net.ninputs);

     gsl_matrix_set_zero(net.posprobs);

  

#pragma omp parallel for
  for(int thread=0;thread<(int) net.maxnumthreads;++thread){


    //add hidden bias
    for(int i=0;i<(int) net.sizesblocks[thread];++i)

      gsl_matrix_set_row (&net.blockposprobs[thread].matrix,i,&net.hidbias.vector);


    //multiply weights
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
		   1.0, &net.blocktdata[thread].matrix, &net.pureweights.matrix,
		   1.0, &net.blockposprobs[thread].matrix);


    logistic (&net.blockposprobs[thread].matrix);
	
  }

    
}




void logistic (gsl_matrix * m,size_t nrows,size_t ncols){

  for(unsigned i=0;i<nrows;++i)
    for(unsigned j=0;j<ncols;++j)
      gsl_matrix_set(m,i,j,1/(1+exp((-1)*gsl_matrix_get(m,i,j))));

}
