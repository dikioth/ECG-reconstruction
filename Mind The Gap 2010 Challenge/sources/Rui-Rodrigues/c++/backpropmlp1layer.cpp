
     /* backpropmlp1layer.cpp - train the waits of an mlp with one hidden layer using backprop algorithm
        Copyright (C) 2010  Rui Rodrigues <rapr@fct.unl.pt>
        This software is released under the terms of the GNU General
        Public License (http://www.gnu.org/copyleft/gpl.html).
     */



//---------------------------------------------------------------------------------------------------
// from the 1rst hidden layer of a set of signals to the 2nd hidden layer of target signal
//---------------------------------------------------------------------------------------------------

#include <string>
#include <sys/time.h>
#include <vector>
#include <iostream>
#include <fstream>

using namespace std;
#include "gradgsl.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_permutation.h>
#include "netdimsandfilenames.h"



void writegslmatriz(const char* filename,gsl_matrix*m);
void readgslmatriz(const char* filename,gsl_matrix*&m);
//in iogslvectormatrix.cpp


void checkfstream(ifstream& file_io,const char* filename);
//in checkfstream.cpp


void read_datafile(ifstream&in,netdimsandfilenames& A);
//in netdimsandfilenames.cpp


//void convertmatrixtolinevectorcway(const matrix&original,matrix&linevector);
//down in this file



void savegslvectormatrix(FILE* writetofile, gsl_vector* v,vector<int>&sizes);
//iogslvectormatrices.cpp

void readgslvectormatrix(FILE* readfromfile, gsl_vector* v,vector<int>&sizes);
void readgslvectorasmatriz(const char* filename,gsl_vector*v);
//iogslvectormatrices.cpp

void useblacklist(gsl_matrix * &inputdata,const char*blacklistfile);
//in blacklist.cpp



// -------------------------------------------------------------------


const string start="start";

const string cont="cont";

const string startperceptron="startperceptron";

const string blacklist_use="useblacklist";

// ----------------------------------------------------------------------------------------
//-----------------------CONFIGURE-----------------------------------------------------

const size_t batchsize=500;

const size_t numepochs=200;

//-----------------------------------------------------------------------------------


int main(int argc, char ** argv){


  try{

if(argc<5){

  cout<<" must be called with argument start, startperceptron or  cont and after signal1 then signal2 and folder name. Optionaly there is an extra argument: useblacklist !"<<endl;
    exit(0);
  }



  int a=0;
  
  if(start.compare(argv[1])==0)
    a=1;

  if(cont.compare(argv[1])==0)
    a=2;

  if(startperceptron.compare(argv[1])==0)
    a=3;


  if(a==0){

    cout<<"first argument must be start,  or cont"<<endl;
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




 //get filenames to extract dimensions and else--


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

  int ninputs=A1.nhidden0, nhidden0=A1.nhidden1, noutputs=A2.nhidden1; 

  size_t numhidlayers=1;

  int dimensions[]={ninputs,nhidden0,noutputs};
  //--------------------------------------------------------------------




  //----------------------------------------------------
  //load training data


  cout<<"loading training data"<<endl;



  //matrix inputdata, outputdata;

  gsl_matrix * inputdata,* outputdata;



  cout<<"loading inputdata from "<<A1.autoencoderfirstlayerdatafile.c_str()<<endl;
 

  readgslmatriz(A1.autoencoderfirstlayerdatafile.c_str(),inputdata);


  cout<<"loading outputdata from "<<A2.autoencodersecondlayerdatafile.c_str()<<endl;
 

  readgslmatriz(A2.autoencodersecondlayerdatafile.c_str(),outputdata); 

  if(inputdata->size1!=outputdata->size1){
    cout<<"number of patches for input and outputdata is not the same!"<<endl;
    exit(1);
  }

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



 
  //----------------------------------------------------------





  //for backprop

  parametersgradgsl td(numhidlayers,dimensions,batchsize);  



  size_t npatches=inputdata->size1;

  size_t numbatches=npatches/batchsize;



  //gsl random number generator

  gsl_rng *r = gsl_rng_alloc(gsl_rng_taus2);
  int long seed=time (NULL) * getpid();
  gsl_rng_set(r,seed);




  //load weights ------------------------------------------------------


  //file for mlp1layerweights
  string f="../";
  f.append(cc);
  f.append("/mlp1layers_");
  f.append(signal1);
  f.append("_");
  f.append(signal2);
  f.append(".txt");

  int nrows[]={ninputs+1,nhidden0+1};
  int ncols[]={nhidden0,noutputs};

  gsl_vector * minimumweights=gsl_vector_alloc (td.tamtotal);

  cout<<"loading weights"<<endl;

  if((a==1)||(a==3)){
     
    //even if we only need a small parte we must load all the weights from 1st signal autoencoder
    

    int numcoef=(A1.patchsize*A1.nsignals+1)*A1.nhidden0+(A1.nhidden0+1)*A1.nhidden1
      +(A1.nhidden1+1)*A1.nhidden0+(A1.nhidden0+1)*(A1.patchsize*A1.nsignals);

    gsl_vector* aux1=gsl_vector_alloc(numcoef);

    vector<int> sizes;

    FILE * readweights = fopen(A1.backpropautoencodercoefficientsfile.c_str(), "r");

    readgslvectormatrix(readweights, aux1,sizes);

    int coef2ndmatrixstart=(A1.patchsize*A1.nsignals+1)*A1.nhidden0;
    int numcoef2ndmatrix=(A1.nhidden0+1)*A1.nhidden1;

    gsl_vector_const_view aux2=gsl_vector_const_subvector(aux1, coef2ndmatrixstart,numcoef2ndmatrix);

    gsl_vector_view aux3=gsl_vector_subvector(minimumweights, 0,numcoef2ndmatrix);

    gsl_vector_memcpy (&aux3.vector,&aux2.vector);
    
    if(a==1){

    //for the matrix from the hidden layer to the output the initial weights are random generated;

    //matrix randomweights(1,(A1.nhidden1+1)*A2.nhidden1);
    //randomweights=randn(nnormal,1,(A1.nhidden1+1)*A2.nhidden1);

    for(int i=0;i<(A1.nhidden1+1)*A2.nhidden1;++i)
      //gsl_vector_set(minimumweights,numcoef2ndmatrix+1,randomweights(0,i));
      gsl_vector_set(minimumweights,numcoef2ndmatrix+1,gsl_ran_gaussian(r,1.0));
    }
    else if(a==3){

      //load weights from perceptron

      
        //file f perceptronweights
      string p="../";
      p.append(cc);
      p.append("/perceptron_");
      p.append(signal1);
      p.append("_");
      p.append(signal2);
      p.append(".txt");


      gsl_vector_view aux4=gsl_vector_subvector(minimumweights, 
						numcoef2ndmatrix,
						nrows[1]*ncols[1]);


      //debug 
//      cout<<"before loading perceptron weights the norm of this parte of weights vector is "<< gsl_blas_dnrm2(&aux4.vector)<<endl;

      readgslvectorasmatriz(p.c_str(),&aux4.vector);


      //debug 
      //    cout<<"after loading perceptron weights the norm of this parte of weights vector is "<< gsl_blas_dnrm2(&aux4.vector)<<endl;



    }
  }

  else if(a==2){

   vector<int> sizes;

    FILE * readweights = fopen(f.c_str(), "r");

    readgslvectormatrix(readweights, minimumweights,sizes);
    
  }


  //end loading weights----------------------------------------

 
  //compute error with initialweights

  {
    cout<<"computing initial error rate"<<endl;


    int num=npatches;

    parametersfwdgsl parameters(numhidlayers,dimensions,num);  

    gsl_matrix_memcpy(&parameters.reallayerdata[0].matrix,inputdata);

    double error_rate=just_compute_error_gsl_vislogistic(minimumweights,&parameters,outputdata);

    cout<<"error rate before training is "<<error_rate*error_rate/npatches<<endl;


  }
 




  //   begin optimization----------------------------------------------------------------------------


  //have many test trials

  int permutation[npatches];

  for(int i=0;i<(int) npatches;++i)
    permutation[i]=i;

  gsl_matrix *permutedinputdata=gsl_matrix_calloc(npatches,ninputs+1);

  gsl_matrix *permutedoutputdata=gsl_matrix_alloc(npatches,noutputs);


  gsl_vector_view auxrow;

  gsl_matrix_view auxoutputview;

  gsl_matrix_view auxinputview;


  //permutedinputdata last column of ones
  gsl_vector_view v=gsl_matrix_column (permutedinputdata,ninputs);
  gsl_vector_add_constant (&v.vector, 1.0);


  gsl_matrix_view real_permutedinputdata=gsl_matrix_submatrix(permutedinputdata,0,0,
							      npatches,ninputs);


  //debug
  //numbatches=2;


  for(int epoch=0;epoch<(int) numepochs;++epoch){


    gsl_ran_shuffle (r, permutation, npatches, sizeof(int));



    for(int i=0;i<(int) npatches;++i){

  
      //inputdata
     
      auxrow=gsl_matrix_row (inputdata, permutation[i]);
     
      gsl_matrix_set_row (&real_permutedinputdata.matrix,i,&auxrow.vector);


      //outputdata

      auxrow=gsl_matrix_row (outputdata, permutation[i]);
      
      gsl_matrix_set_row (permutedoutputdata,i,&auxrow.vector);

    }





    for(int batch=0;batch<(int) numbatches;++batch){


      auxinputview=gsl_matrix_submatrix (permutedinputdata, batch*batchsize, 0, 
					 batchsize,ninputs+1);
      td.layerdata[0]=&auxinputview.matrix;
      
      auxoutputview=gsl_matrix_submatrix (permutedoutputdata, batch*batchsize, 0, 
					  batchsize,noutputs);
      td.batchoutputdata=&auxoutputview.matrix;


      //initialing GSL-minimizer----------------------------

      const gsl_multimin_fdfminimizer_type *T;

      gsl_multimin_fdfminimizer *s;

      gsl_multimin_function_fdf my_func;

      my_func.n = td.tamtotal;



      my_func.f = compute_error_gsl_vislogistic ;

      my_func.df = gradientgsl_vislogistic ;

      my_func.fdf = errorandgrad_vislogistic;
    

      my_func.params = &td;

      T = gsl_multimin_fdfminimizer_conjugate_pr;

      s = gsl_multimin_fdfminimizer_alloc (T, td.tamtotal);

      gsl_multimin_fdfminimizer_set (s, &my_func, minimumweights, 0.01, 1e-4);

      //-------


      //iterate minimizer

      size_t iter = 0;

      size_t maxiterations=2;

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

      printf("epoch %4i batch %4i  error rate %f \r", epoch, batch,(s->f)*(s->f)/batchsize);
      fflush(stdout);


      gsl_vector_memcpy (minimumweights,s->x);

      gsl_multimin_fdfminimizer_free (s);

    }

  }

  gsl_matrix_free(permutedinputdata);

  gsl_matrix_free(permutedoutputdata);

  gsl_rng_free (r);

  //end of optimization------------------------------------------

  


  //write new weights in a file-------------------------------------

  int matricesdimensions[]={2,dimensions[0]+1,dimensions[1],dimensions[1]+1,dimensions[2]};

  vector<int>  matdimensions(matricesdimensions,matricesdimensions+5);




  FILE * writeweights = fopen(f.c_str(), "w");

    
  savegslvectormatrix(writeweights, minimumweights,matdimensions);

  fclose(writeweights);


  
   //-------------------------computer error rate after training------------

  {

    parametersfwdgsl parameters(numhidlayers,dimensions,npatches); 


    gsl_matrix_memcpy(&parameters.reallayerdata[0].matrix,inputdata);

    double error_rate=just_compute_error_gsl_vislogistic(minimumweights,&parameters,outputdata);

    cout<<"error rate after training is "<<error_rate*error_rate/npatches<<endl;
    
  }
  

  gsl_vector_free(minimumweights); 

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
