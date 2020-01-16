
     /* trainboltzmannperceptron.cpp - train perceptron in boltzmann machine way (random zero one output)
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

#include "gradgsl.h"
#include "netdimsandfilenames.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_permutation.h>



void checkfstream(ofstream& file_io,const char* filename);
//in checkfstream.cpp

void checkfstream(ifstream& file_io,const char* filename);
//in checkfstream.cpp


void read_datafile(ifstream&in,netdimsandfilenames& A);
//in netdimsandfilenames.cpp


void logistic (gsl_matrix * m,int nrows,int ncols);
//in parallelgradgsl.cpp

void writegslmatriz(const char* filename,gsl_matrix*m);
void readgslmatriz(const char* filename,gsl_matrix*&m);
//in iogslvectormatrix.cpp


void useblacklist(gsl_matrix * &inputdata,const char*blacklistfile);
//in blacklist.cpp



const string start="start";

const string cont="cont";

const string blacklist_use="useblacklist";


// ----------------------------------------------------------------------------------------
//-----------------------CONFIGURE-----------------------------------------------------

const size_t batchsize=500;

const size_t numepochs=100;

const double epsilonweights=0.1;

const double epsilonbias=0.1;

const double momentum=0.5;

const double weightscost=0.002;

//-----------------------------------------------------------------------------------


int main(int argc, char ** argv){


  try{

if(argc<5){

    cout<<" must be called with argument start or cont, after signal1 then signal2 and at last folder name. Optionaly thereis an extra argument: useblacklist !"<<endl;

    exit(0);
  }


  int a=0;
  
  if(start.compare(argv[1])==0)
    a=1;

  if(cont.compare(argv[1])==0)
    a=2;


  if(a==0){

    cout<<"first argument must be start or cont"<<endl;
    exit(0);
  }


  int auxb=0;


  if(argc==6)
    if(blacklist_use.compare(argv[5])==0)
      auxb=1;
    else{
      cout<<"wrong last argument!";
      exit(1);
    }
  else;



  //time -----------------
  struct timeval start, end;
  gettimeofday(&start, NULL);
  //----------------------



 //get filename to extract dimensions and else--


  //signal1

  string signal1=argv[2];
  string cc=argv[4];
  string b1=signal1;
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
  

  //signal2


  string signal2=argv[3];
  string b2=signal2; 
  b2.append(".txt");
  string d2=folder;
  d2.append(b2);

  cout<<"reading configuration data from "<<d2.c_str()<<endl;

  netdimsandfilenames A2;

  ifstream reading2(d2.c_str());

  read_datafile(reading2,A2);

  checkfstream(reading2,d2.c_str());

  reading2.close();





  //----------------------------

  int ninputs=A1.nhidden1;

  int noutputs=A2.nhidden1;

  //--------------------------------------------------------------------


 //----------------------------------------------------
  //load training data

  gsl_matrix * inputdata,* outputdata;
 
  readgslmatriz(A1.autoencodersecondlayerdatafile.c_str(),inputdata);

  readgslmatriz(A2.autoencodersecondlayerdatafile.c_str(),outputdata); 
  


  if((int) inputdata->size2!=ninputs){
    cout<<"inputdata is not compatible with ninputs!"<<endl;
    exit(1);
  }

  if((int) outputdata->size2!=noutputs){
    cout<<"outputdata is not compatible with noutputs!"<<endl;
    exit(1);
  }


  if(auxb==1){

    string blacklistfile=folder;
    blacklistfile.append(signal1);
    blacklistfile.append("_blacklist.txt");    

    useblacklist(inputdata,blacklistfile.c_str());

     useblacklist(outputdata,blacklistfile.c_str());   
  }
 


  size_t npatches=inputdata->size1;

  size_t numbatches=npatches/batchsize;


 

  //----------------------------------------------------------



  //gsl random number generator

  gsl_rng *r = gsl_rng_alloc(gsl_rng_taus2);

  int long seed=time (NULL) * getpid();

  gsl_rng_set(r,seed);


 



  //file for perceptronweights
  string f="../";
  f.append(cc);
  f.append("/perceptron_");
  //  string signal1=argv[2];
  //string signal2=argv[3];
  f.append(signal1);
  f.append("_");
  f.append(signal2);
  f.append(".txt");


  //load weights 

  gsl_matrix * weights=gsl_matrix_alloc (ninputs+1,noutputs);



  if(a==1)

    //random weights

    for(int i=0; i<ninputs+1;++i)
      for(int j=0;j<noutputs;++j)
	gsl_matrix_set(weights,i,j,gsl_ran_gaussian(r,0.01));

   
  else if(a==2)

    //load weights from file

     readgslmatriz(f.c_str(),weights);
  



  gsl_matrix_view pureweights=gsl_matrix_submatrix (weights, 0, 0, ninputs, noutputs);


  gsl_vector_view bias=gsl_matrix_row (weights,ninputs);





 //-------------------------computer error rate before training------------


  gsl_vector*vectorimagedata=gsl_vector_alloc(npatches*noutputs);
  
  gsl_matrix_view imagedata=gsl_matrix_view_vector(vectorimagedata,npatches,noutputs);
    
  //insert bias on matrix

  for(int i=0;i<(int) npatches;++i)
    gsl_matrix_set_row (&imagedata.matrix,i,&bias.vector);



  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
		 1.0, inputdata, &pureweights.matrix,
		 1.0, &imagedata.matrix);
  

  logistic (&imagedata.matrix,npatches,noutputs);


  gsl_matrix_sub (&imagedata.matrix,outputdata);


  double error=gsl_blas_dnrm2 (vectorimagedata);

  
  cout<<"square error rate by patch before training is "<<(error*error)/npatches<<endl;


  //---------------------------------------------------------------------




  int permutation[npatches];

  for(int i=0;i<(int) npatches;++i)
    permutation[i]=i;


  gsl_matrix* permuted_inputdata=gsl_matrix_alloc(npatches,ninputs);

  gsl_matrix* permuted_outputdata=gsl_matrix_alloc(npatches,noutputs);


  gsl_vector_view auxrow;
 

  gsl_matrix_view batchinputdata, batchoutputdata;


  gsl_matrix *product=gsl_matrix_alloc(batchsize,noutputs);


  gsl_vector * ones=gsl_vector_calloc(batchsize);

  gsl_vector_add_constant (ones,1.0);

  
  gsl_matrix * deltaweights=gsl_matrix_alloc(ninputs,noutputs);


  gsl_vector * deltabias=gsl_vector_alloc(noutputs);



  for(int epoch=0;epoch<(int) numepochs;++epoch){


    //randomly permute different segments of data---------------------


    gsl_ran_shuffle (r, permutation, npatches, sizeof(int));
    

    for(int i=0;i<(int) npatches;++i){
 
 
      //inputdata
     
      auxrow=gsl_matrix_row (inputdata, permutation[i]);

      
      gsl_matrix_set_row (permuted_inputdata,i,&auxrow.vector);


      //outputdata

      auxrow=gsl_matrix_row (outputdata, permutation[i]);
      
      gsl_matrix_set_row (permuted_outputdata,i,&auxrow.vector);

    }





    //-----------------------------------------------------------------



    for(int batch=0;batch<(int) numbatches;++batch){


      //inputbatchdata


      batchinputdata=gsl_matrix_submatrix (permuted_inputdata, batch*batchsize,
					   0, batchsize, ninputs);


      batchoutputdata=gsl_matrix_submatrix (permuted_outputdata, batch*batchsize,
					   0, batchsize, noutputs);


      //insert bias on matrix

      for(int i=0;i<(int) batchsize;++i)
	gsl_matrix_set_row (product,i,&bias.vector);


      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
                       1.0, &batchinputdata.matrix, &pureweights.matrix,
                       1.0, product);


      logistic (product,batchsize,noutputs);


      //generate uniform random data and compare with product
      //if the product its bigger it gets one otherwise zero

      for(int i=0;i<(int) batchsize;++i)
	for(int j=0;j<noutputs;++j)
	  if (gsl_matrix_get(product,i,j)>gsl_rng_uniform (r))	  
	    gsl_matrix_set(product,i,j,1.0);
	  else
	    gsl_matrix_set(product,i,j,0.0);
       

      gsl_matrix_scale (product, -1.0);

      gsl_matrix_add (product,&batchoutputdata.matrix );


      //deltabias and new bias

      gsl_blas_dgemv (CblasTrans, epsilonbias/batchsize,product, 
		      ones, momentum, deltabias);

      gsl_vector_add (&bias.vector,deltabias);


      //deltaweights and new weights

      gsl_blas_dgemm(CblasTrans, CblasNoTrans,
                       epsilonweights/batchsize, &batchinputdata.matrix, product,
                       momentum, deltaweights);
      
 
      //compensate for weight cost

      gsl_matrix_scale (&pureweights.matrix, 1.0-weightscost);

      gsl_matrix_add (&pureweights.matrix,deltaweights);


      gsl_vector_view diff=gsl_vector_view_array (product->data, batchsize*noutputs);

      error=gsl_blas_dnrm2 (&diff.vector);
      
      
      printf("epoch %4i batch %4i  square error rate per patch  %e \r", epoch, batch,error*error/batchsize);
      fflush(stdout);

    }




  }


  gsl_vector_free(deltabias);

  gsl_matrix_free(deltaweights);

  gsl_vector_free(ones);

  gsl_matrix_free(product);

  gsl_matrix_free(permuted_inputdata);

  gsl_matrix_free(permuted_outputdata);
 
  gsl_rng_free (r); 

  //save weights


  writegslmatriz(f.c_str(),weights);





  //gsl_matrix_free(batchoutputdata);



 //-------------------------computer error rate after training------------

     
  //insert bias on matrix

  for(int i=0;i<(int) npatches;++i)
    gsl_matrix_set_row (&imagedata.matrix,i,&bias.vector);



  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
		 1.0, inputdata, &pureweights.matrix,
		 1.0, &imagedata.matrix);
  

  logistic (&imagedata.matrix,npatches,noutputs);


  gsl_matrix_sub (&imagedata.matrix,outputdata);


  error=gsl_blas_dnrm2 (vectorimagedata);

  
  cout<<"square error rate by patch after training is "<<(error*error)/npatches<<endl;

    

  //---------------------------------------------------------------------


  gsl_vector_free(vectorimagedata);

  gsl_matrix_free(weights);

  gsl_matrix_free(inputdata);

  gsl_matrix_free(outputdata);




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



