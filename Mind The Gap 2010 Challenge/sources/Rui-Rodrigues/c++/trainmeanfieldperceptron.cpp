
     /* trainmeanfieldperceptron.cpp - train perceptron in meanfield boltzmann machine 
	way (deterministic real valued logistic outputs)
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
//in ../geral/checkfstream.cpp

void checkfstream(ifstream& file_io,const char* filename);
//in ../geral/checkfstream.cpp


void read_datafile(ifstream&in,netdimsandfilenames& A);
//in netdimsandfilenames.cpp


void logistic (gsl_matrix * m,int nrows,int ncols);
//in parallelgradgsl.cpp

void writegslmatriz(const char* filename,gsl_matrix*m);
void readgslmatriz(const char* filename,gsl_matrix*&m);
void readgslvectorasmatriz(const char* filename,gsl_vector*v);
//in iogslvectormatrix.cpp


void useblacklist(gsl_matrix * &inputdata,const char*blacklistfile);
//in blacklist.cpp



double computeerrorperceptron(const gsl_vector *vectorweights, void *params);
void computegradientperceptron(const gsl_vector *vectorweights, void *params, gsl_vector *gradient);
void  computegradanderror(const gsl_vector *vectorweights, void *params, double *error, gsl_vector *gradient);

// ----------------------------------------------------------------------------------------



//-----------------------CONFIGURE-----------------------------------------------------

const size_t batchsize=500;

const size_t numepochs=100;

const double epsilonweights=0.1;

const double epsilonbias=0.1;

const double momentum=0.5;

const double weightscost=0.002;

//-----------------------------------------------------------------------------------


const string blacklist_use="useblacklist";


struct inputoutputdata{

  gsl_matrix * ptinputdata, *ptoutputdata, *auxgrad;
  //auxgrad is used to compute grad and must be previously allocated with
  //dimensions (ptinputdata->size2,ptoutputdata->size2)


  gsl_vector* vectorimagedata, *ones;
  //vectorimagedata must be previously allocated with size (ptinputdata->size1)*(ptoutputdata->size2)
  //ones is used to compute grad and must be previously allocated with size (ptinputdata->size1)
  //     and filled with ones

  double weightscost;

};



int main(int argc, char ** argv){


  try{

if(argc<4){

    cout<<" must be called with argument  signal1 then signal2 and at last folder name. Optionaly thereis an extra argument: useblacklist !"<<endl;

    exit(0);
  }


 int auxb=0;

  if(argc==5)
    if(blacklist_use.compare(argv[4])==0)
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

  string signal1=argv[1];
  string cc=argv[3];
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
  string signal2=argv[2];
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

  unsigned ninputs=A1.nhidden1;

  unsigned noutputs=A2.nhidden1;

  //--------------------------------------------------------------------


  
 //----------------------------------------------------
  //load training data

  gsl_matrix * inputdata,* outputdata;
 
  readgslmatriz(A1.autoencodersecondlayerdatafile.c_str(),inputdata);

  readgslmatriz(A2.autoencodersecondlayerdatafile.c_str(),outputdata); 
  


  if(inputdata->size2!=ninputs){
    cout<<"inputdata is not compatible with ninputs!"<<endl;
    exit(1);
  }

  if(outputdata->size2!=noutputs){
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

  unsigned long seed=time (NULL) * getpid();

  gsl_rng_set(r,seed);



 



  //file for perceptronweights
  string f="../";
  f.append(cc);
  f.append("/perceptron_");
  f.append(signal1);
  f.append("_");
  f.append(signal2);
  f.append(".txt");


  //load weights 

  gsl_vector * vectorweights=gsl_vector_alloc((ninputs+1)*noutputs);




  readgslvectorasmatriz(f.c_str(),vectorweights);






  inputoutputdata params;

  
 //-------------------------computer error rate before training------------

  gsl_vector *vectorimage=gsl_vector_alloc(npatches*noutputs);

  params.ptinputdata=inputdata;

  params.ptoutputdata=outputdata;

  params.vectorimagedata=vectorimage;


  double error=computeerrorperceptron(vectorweights,&params);
  
  cout<<"square error rate by patch before training is "<<(error*error)/npatches<<endl;


  //---------------------------------------------------------------------



  unsigned permutation[npatches];

  for(unsigned i=0;i<npatches;++i)
    permutation[i]=i;


  gsl_matrix* permuted_inputdata=gsl_matrix_alloc(npatches,ninputs);

  gsl_matrix* permuted_outputdata=gsl_matrix_alloc(npatches,noutputs);

  gsl_matrix_view batchinputdata, batchoutputdata;

  gsl_vector * ones=gsl_vector_calloc(batchsize);

  gsl_vector_add_constant (ones,1.0);

  gsl_vector_view auxrow;  

  gsl_vector* batchvectorimage=gsl_vector_alloc(batchsize*noutputs);

  gsl_matrix *auxgrad=gsl_matrix_alloc(ninputs,noutputs);



  for(unsigned epoch=0;epoch<numepochs;++epoch){


    //randomly permute different segments of data---------------------


    gsl_ran_shuffle (r, permutation, npatches, sizeof(unsigned));
    

    for(unsigned i=0;i<npatches;++i){
 
 
      //inputdata
     
      auxrow=gsl_matrix_row (inputdata, permutation[i]);

      
      gsl_matrix_set_row (permuted_inputdata,i,&auxrow.vector);


      //outputdata

      auxrow=gsl_matrix_row (outputdata, permutation[i]);
      
      gsl_matrix_set_row (permuted_outputdata,i,&auxrow.vector);

    }

  //-----------------------------------------------------------------



    for(unsigned batch=0;batch<numbatches;++batch){


      //inputbatchdata


      batchinputdata=gsl_matrix_submatrix (permuted_inputdata, batch*batchsize,
					   0, batchsize, ninputs);


      batchoutputdata=gsl_matrix_submatrix (permuted_outputdata, batch*batchsize,
					   0, batchsize, noutputs);
      

      params.ptinputdata=&batchinputdata.matrix;


      params.ptoutputdata=&batchoutputdata.matrix;


      params.vectorimagedata=batchvectorimage;


      params.auxgrad=auxgrad;


      params.ones=ones;

      params.weightscost=weightscost;
      
      //initialing GSL-minimizer----------------------------

      const gsl_multimin_fdfminimizer_type *T;

      gsl_multimin_fdfminimizer *s;

      gsl_multimin_function_fdf my_func;

      my_func.n = (ninputs+1)*noutputs;

      my_func.f = computeerrorperceptron;

      my_func.df = computegradientperceptron;

      my_func.fdf = computegradanderror;

      my_func.params = &params;

      T = gsl_multimin_fdfminimizer_conjugate_pr;

      s = gsl_multimin_fdfminimizer_alloc (T, (ninputs+1)*noutputs);

      gsl_multimin_fdfminimizer_set (s, &my_func,vectorweights, 0.001, 1e-4);

      //-------


    //iterate minimaizer

      size_t iter = 0;

      size_t maxiterations=3;

      int status;

      do
	{
	  iter++;
	  status = gsl_multimin_fdfminimizer_iterate (s);
     
	  //test if everything is ok
	  if (status)
	    break;
     
	  status = gsl_multimin_test_gradient (s->gradient, 1e-3);

	  //printf ("%5d %10.5f\n", iter,s->f);
         
	}
      while (status == GSL_CONTINUE && iter < maxiterations);

      printf("epoch %4i batch %4i squared error rate by patch  %e \r", epoch, batch,(s->f)*(s->f)/batchsize);
      fflush(stdout);


      gsl_vector_memcpy (vectorweights,s->x);

      gsl_multimin_fdfminimizer_free (s);


    }

  }


  gsl_matrix_free(permuted_inputdata);

  gsl_matrix_free(permuted_outputdata);

  gsl_vector_free(ones);

  gsl_vector_free(batchvectorimage);

  gsl_matrix_free(auxgrad);


  //save new weights
  gsl_matrix_view  weightsmatrix=gsl_matrix_view_vector(vectorweights,ninputs+1,noutputs);

  writegslmatriz(f.c_str(),&weightsmatrix.matrix);


 //-------------------------computer error rate after training------------

  params.ptinputdata=inputdata;

  params.ptoutputdata=outputdata;

  params.vectorimagedata=vectorimage;


  error=computeerrorperceptron(vectorweights,&params);
  
  cout<<"square error rate by patch after training is "<<(error*error)/npatches<<endl;


  //---------------------------------------------------------------------



  gsl_vector_free(vectorimage);

  gsl_vector_free(vectorweights);


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









//functions for the minimizer

double computeerrorperceptron(const gsl_vector *vectorweights, void *params){



  inputoutputdata * ptdata=(inputoutputdata*) params;

  gsl_matrix*inputdata=ptdata->ptinputdata;

  gsl_matrix* outputdata=ptdata->ptoutputdata;

  gsl_vector*vectorimagedata=ptdata->vectorimagedata;



  unsigned batchsize=inputdata->size1;

  unsigned ninputs=inputdata->size2;

  unsigned noutputs=outputdata->size2;



  if(vectorweights->size!=(ninputs+1)*noutputs)    
    throw "input, output data and weights dimensions are not compatible!";


  if(vectorimagedata->size!=batchsize*noutputs)
    throw "vectorimagedata dimensions are not correct!";



  gsl_matrix_const_view pureweights=gsl_matrix_const_view_vector (vectorweights, ninputs, noutputs);

  gsl_vector_const_view bias=gsl_vector_const_subvector (vectorweights,ninputs*noutputs,noutputs);

  gsl_matrix_view imagedata=gsl_matrix_view_vector(vectorimagedata,batchsize,noutputs);



  //insert bias on matrix

  for(unsigned i=0;i<batchsize;++i)
    gsl_matrix_set_row (&imagedata.matrix,i,&bias.vector);
  
  
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
		 1.0, inputdata, &pureweights.matrix,
		 1.0, &imagedata.matrix);


  logistic (&imagedata.matrix,batchsize,noutputs);


  gsl_matrix_sub (&imagedata.matrix,outputdata);


  double error=gsl_blas_dnrm2 (vectorimagedata);

  return error;

}



void computegradientperceptron(const gsl_vector *vectorweights, void *params, gsl_vector *gradient){

  
  inputoutputdata * ptdata=(inputoutputdata*) params;

  gsl_matrix*inputdata=ptdata->ptinputdata;

  gsl_matrix* outputdata=ptdata->ptoutputdata;

  gsl_matrix* auxgrad=ptdata->auxgrad;

  gsl_vector*vectorimagedata=ptdata->vectorimagedata;

  gsl_vector*ones=ptdata->ones;

  double weightscost=ptdata->weightscost;
 

  unsigned batchsize=inputdata->size1;

  unsigned ninputs=inputdata->size2;

  unsigned noutputs=outputdata->size2;


  if(vectorweights->size!=(ninputs+1)*noutputs)    
    throw "input, output data and weights dimensions are not compatible!";

  if(gradient->size!=(ninputs+1)
*noutputs)    
    throw "gradient and weights dimensions are not compatible!";
  

  if((auxgrad->size1!=ninputs)||(auxgrad->size2!=noutputs))    
    throw "auxgrad dimensions are not correct!";

  if(vectorimagedata->size!=batchsize*noutputs)
    throw "vectorimagedata dimensions are not correct!";

  if(ones->size!=batchsize)
    throw "ones dimension is not correct!";

  gsl_matrix_const_view pureweights=gsl_matrix_const_view_vector (vectorweights, ninputs, noutputs);

  gsl_vector_const_view bias=gsl_vector_const_subvector (vectorweights,ninputs*noutputs,noutputs);
    
  gsl_matrix_view imagedata=gsl_matrix_view_vector(vectorimagedata,batchsize,noutputs);



  //insert bias on matrix

  for(unsigned i=0;i<batchsize;++i)
    gsl_matrix_set_row (&imagedata.matrix,i,&bias.vector);
  
  
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
		 1.0, inputdata, &pureweights.matrix,
		 1.0, &imagedata.matrix);


  logistic (&imagedata.matrix,batchsize,noutputs);

  gsl_matrix_sub (&imagedata.matrix,outputdata);


  //pure weights grad

  gsl_matrix_view puregrad=gsl_matrix_view_vector(gradient,ninputs,noutputs);


  gsl_blas_dgemm(CblasTrans, CblasNoTrans,1.0,
		 inputdata,&imagedata.matrix,0.0,&puregrad.matrix);

  //weightscost
  gsl_matrix_memcpy (auxgrad,&pureweights.matrix);

  gsl_matrix_scale (auxgrad,weightscost);

  gsl_matrix_sub (&puregrad.matrix,auxgrad);


  //bias grad

  gsl_vector_view deltabias=gsl_vector_subvector (gradient,ninputs*noutputs,noutputs);

  gsl_blas_dgemv (CblasTrans, 1.0,&imagedata.matrix, 
		  ones, 0.0, &deltabias.vector);

}


void  computegradanderror(const gsl_vector *vectorweights, void *params, 
             double *error, gsl_vector *gradient) {



  inputoutputdata * ptdata=(inputoutputdata*) params;

  gsl_matrix*inputdata=ptdata->ptinputdata;

  gsl_matrix* outputdata=ptdata->ptoutputdata;

  gsl_matrix* auxgrad=ptdata->auxgrad;

  gsl_vector*vectorimagedata=ptdata->vectorimagedata;

  gsl_vector*ones=ptdata->ones;

  double weightscost=ptdata->weightscost;



  unsigned batchsize=inputdata->size1;

  unsigned ninputs=inputdata->size2;

  unsigned noutputs=outputdata->size2;


  if(vectorweights->size!=(ninputs+1)*noutputs)    
    throw "input, output data and weights dimensions are not compatible!";

  if(gradient->size!=(ninputs+1)*noutputs)    
    throw "gradient and weights dimensions are not compatible!";
  

  if((auxgrad->size1!=ninputs)||(auxgrad->size2!=noutputs))    
    throw "auxgrad dimensions are not correct!";

  if(vectorimagedata->size!=batchsize*noutputs)
    throw "vectorimagedata dimensions are not correct!";

  if(ones->size!=batchsize)
    throw "ones dimension is not correct!";

  gsl_matrix_const_view pureweights=gsl_matrix_const_view_vector (vectorweights, ninputs, noutputs);

  gsl_vector_const_view bias=gsl_vector_const_subvector (vectorweights,ninputs*noutputs,noutputs);

  gsl_matrix_view imagedata=gsl_matrix_view_vector(vectorimagedata,batchsize,noutputs);


  //insert bias on matrix

  for(unsigned i=0;i<batchsize;++i)
    gsl_matrix_set_row (&imagedata.matrix,i,&bias.vector);
  
  
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
		 1.0, inputdata, &pureweights.matrix,
		 1.0, &imagedata.matrix);


  logistic (&imagedata.matrix,batchsize,noutputs);


  gsl_matrix_sub (&imagedata.matrix,outputdata);


  *error=gsl_blas_dnrm2 (vectorimagedata);



  //pure weights grad

  gsl_matrix_view puregrad= gsl_matrix_view_vector(gradient,ninputs,noutputs);


  gsl_blas_dgemm(CblasTrans, CblasNoTrans,1.0,
		 inputdata,&imagedata.matrix,0.0,&puregrad.matrix);

  //weightscost
  gsl_matrix_memcpy (auxgrad,&pureweights.matrix);

  gsl_matrix_scale (auxgrad,weightscost);

  gsl_matrix_sub (&puregrad.matrix,auxgrad);


  //bias grad

  gsl_vector_view deltabias=gsl_vector_subvector (gradient,ninputs*noutputs,noutputs);

  gsl_blas_dgemv (CblasTrans, 1.0,&imagedata.matrix, 
		  ones, 0.0, &deltabias.vector);



}
