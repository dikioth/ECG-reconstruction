     /* backpropmlp4layers.cpp - train the waits of an mlp with four hidden layers using backprop algorithm
        Copyright (C) 2010  Rui Rodrigues <rapr@fct.unl.pt>
        This software is released under the terms of the GNU General
        Public License (http://www.gnu.org/copyleft/gpl.html).
     */



#include <stdio.h>
#include <string>
#include <sys/time.h>

using namespace std;

#include<vector>
#include <iostream>
#include <fstream>
#include "gradgsl.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "netdimsandfilenames.h"



void checkfstream(ofstream& file_io,const char* filename);
//in ../geral/checkfstream.cpp

void checkfstream(ifstream& file_io,const char* filename);
//in ../geral/checkfstream.cpp

void checkfstream(FILE* pointer,const char* filename);



void read_datafile(ifstream&in,netdimsandfilenames& A);
//in netdimsandfilenames.cpp


void readgslvectormatrix(FILE* readfromfile, gsl_vector* v,vector<int>&sizes);
//in iosgslvectormatrix.cpp

void readgslmatriz(const char* filename,gsl_matrix*&m);
//in iosgslvectormatrix.cpp

void writegslmatriz(const char* filename,gsl_matrix*m);
//save a vector that contains several gslmatrices 
void savegslvectormatrix(FILE* writetofile, gsl_vector* v,vector<int>&sizes);
//in iosgslvectormatrix.cpp

int givesize_gslvector_infile(FILE* readfromfile);
//in iosgslvectormatrix.cpp


void useblacklist(gsl_matrix * &inputdata,const char*blacklistfile);
//in blacklist.cpp



//the following will be use to choose to load initial weights for rbmlogistic or backproprbm weights

const int num_matrices_autoencoder=4;




// -------------------------------------------------------------------



const string startperceptron="startperceptron";

const string startmlp1="startmlp1";

const string cont="cont";

const string blacklist_use="useblacklist";

// ----------------------------------------------------------------------------------------
//-----------------------CONFIGURE-----------------------------------------------------

const int batchsize=500;

const int numepochs=300;

//-----------------------------------------------------------------------------------


int main(int argc, char ** argv){


  try{

if(argc<5){

    cout<<" must be called with argument startperceptron startmlp1 or  cont and after signal1 then signal2  and folder name. Optionaly there is an extra argument: useblacklist !"<<endl;

    exit(0);
  }


  int a=0;
  
  if(startperceptron.compare(argv[1])==0)
    a=1;

  if(cont.compare(argv[1])==0)
    a=2;

  if(startmlp1.compare(argv[1])==0)
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




  //-------------change in case of different number o layers---------------

  int nhidden0=A1.nhidden0, nhidden1=A1.nhidden1, 
    nhidden2=A2.nhidden1, nhidden3=A2.nhidden0;

  int ninputs=A1.nsignals*A1.patchsize, noutputs=A2.patchsize;//only works for a single signal in the output 

  size_t numhidlayers=4;

  int dimensions[]={ninputs,nhidden0,nhidden1,nhidden2,nhidden3,noutputs};

  //--------------------------------------------------------------------


 //----------------------------------------------------
  //load training data


  gsl_matrix * inputdata, *outputdata;

  readgslmatriz(A1.patchdatafile.c_str(),inputdata);
  
  readgslmatriz(A2.patchdatafile.c_str(),outputdata);

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


  //----------load weights---------------------------

  //file for mlp4layerweights
  string f="../";
  f.append(cc);
  f.append("/mlp4layers_");
  f.append(signal1);
  f.append("_");
  f.append(signal2);
  f.append(".txt");



  //for backprop

  parametersgradgsl td(numhidlayers,dimensions,batchsize);  



  size_t npatches=inputdata->size1;

  size_t numbatches=npatches/batchsize;
  


  //load weights 

  gsl_vector * minimumweights=gsl_vector_alloc (td.tamtotal);




  int nrows[]={ninputs+1,nhidden0+1,nhidden1+1,nhidden2+1,nhidden3+1};
  int ncols[]={nhidden0,nhidden1,nhidden2,nhidden3,noutputs};


  if(a==1){

    //perceptron weights file
    string s="../";
    s.append(cc);
    s.append("/perceptron_");
    string signal1=argv[2];
    string signal2=argv[3];
    s.append(signal1);
    s.append("_");
    s.append(signal2);
    s.append(".txt");




    //which part of weights come from which file

    int limits0weights=nrows[0]*ncols[0]+nrows[1]*ncols[1];
    int limits1weights=nrows[2]*ncols[2];
    int limits2weights=nrows[3]*ncols[3]+nrows[4]*ncols[4];


    //first file


    gsl_vector_view aux0=gsl_vector_subvector (minimumweights, 0,limits0weights);
    
    FILE * readweights0 = fopen(A1.backpropautoencodercoefficientsfile.c_str(), "r");
    checkfstream(readweights0, A1.backpropautoencodercoefficientsfile.c_str());


    //read and trash first 9 ints
    int trash1;
    for(int i=0;i<9;++i)
      if(fscanf(readweights0,"%u",&trash1)!=1) return -1;

    int z=gsl_vector_fscanf (readweights0,&aux0.vector);

    if(z){

      cout<<"problem reading weights from file "<<A1.backpropautoencodercoefficientsfile.c_str()<<endl;

      exit(1);
    }

    
    //second file

    gsl_vector_view aux1=gsl_vector_subvector (minimumweights, limits0weights, limits1weights);   
	
    FILE * readweights1 = fopen(s.c_str(), "r");

    z=gsl_vector_fscanf (readweights1,&aux1.vector);

    if(z){

      cout<<"problem reading weights from file "<<s.c_str()<<endl;

      exit(1);
    }


   //third file

    gsl_vector_view aux2=gsl_vector_subvector (minimumweights, limits1weights, limits2weights);   
	
    FILE * readweights2 = fopen(A2.backpropautoencodercoefficientsfile.c_str(), "r");
    checkfstream(readweights2,A2.backpropautoencodercoefficientsfile.c_str());

    //read and trash the first 9 ints
    for(int i=0;i<9;++i)
      if(fscanf(readweights2,"%d",&trash1)!=1) return -1;      


    //count the coefficients corresponding to the first two matrices of second signal autoencoder 
    int start=(A2.nsignals*A2.patchsize+1)*A2.nhidden0+(A2.nhidden0+1)*A2.nhidden1;

    //read and trash 'start' doubles
    double trash2;
    for(int i=0;i<start;++i)
      if(fscanf(readweights2,"%lf",&trash2)!=1) return -1;


    z=gsl_vector_fscanf (readweights2,&aux2.vector);

    if(z){

      cout<<"problem reading weights from file "<<A2.backpropautoencodercoefficientsfile.c_str()<<endl;

      exit(1);
    }


  }
  else if(a==2){

    vector<int> sizes;

    FILE * readweights = fopen(f.c_str(), "r");
    checkfstream(readweights,f.c_str());

    readgslvectormatrix(readweights, minimumweights,sizes);
  }
  else if(a==3){

    //load weights from autoencoders and mlp1

    
    //mlp1 filename
    string mlp1="../";
    mlp1.append(cc);
    mlp1.append("/mlp1layers_");
    mlp1.append(signal1);
    mlp1.append("_");
    mlp1.append(signal2);
    mlp1.append(".txt");

   //which part of weights come from which file

    int limits0weights=nrows[0]*ncols[0];
    int limits1weights=nrows[1]*ncols[1]+nrows[2]*ncols[2];
    int limits2weights=nrows[3]*ncols[3]+nrows[4]*ncols[4];




    //first file
 
  //debug
  cout<<"loading weights from "<<0<<" to  "<<limits0weights<<endl;


    gsl_vector_view aux0=gsl_vector_subvector (minimumweights, 0,limits0weights);
    
    FILE * readweights0 = fopen(A1.backpropautoencodercoefficientsfile.c_str(), "r");
    checkfstream(readweights0,A1.backpropautoencodercoefficientsfile.c_str());

    //read and trash first 9 ints
    int trash1;
    for(int i=0;i<9;++i)
      if(fscanf(readweights0,"%u",&trash1)!=1) return -1;

    int z=gsl_vector_fscanf (readweights0,&aux0.vector);

    if(z){

      cout<<"problem reading weights from file "<<A1.backpropautoencodercoefficientsfile.c_str()<<endl;

      exit(1);
    }

 
       //second file



    gsl_vector_view aux1=gsl_vector_subvector (minimumweights, limits0weights, limits1weights);   
	
    FILE * readweights1 = fopen(mlp1.c_str(), "r");
    checkfstream(readweights1,mlp1.c_str());


    cout<<"reading weights from file "<<mlp1.c_str()<<endl;

    //read and trash first 5 ints
    
    for(int i=0;i<5;++i)
      if(fscanf(readweights1,"%d",&trash1)!=1){
	cout<<"problem reading data from file  "<<mlp1.c_str()<<endl;
	exit(-1);
      };


    z=gsl_vector_fscanf (readweights1,&aux1.vector);

    if(z){

      cout<<"problem reading weights from file "<<mlp1.c_str()<<endl;

      exit(1);
    }


    //third file



    gsl_vector_view aux2=gsl_vector_subvector (minimumweights, limits0weights+limits1weights, limits2weights);   
	
    FILE * readweights2 = fopen(A2.backpropautoencodercoefficientsfile.c_str(), "r");
    checkfstream(readweights2,A2.backpropautoencodercoefficientsfile.c_str());


    //read and trash the first 9 ints
    for(int i=0;i<9;++i)
      if(fscanf(readweights2,"%d",&trash1)!=1){
	cout<<"problem reading data from file  "<<A2.backpropautoencodercoefficientsfile.c_str()<<endl;
	exit(-1);
      };     


    //count the coefficients corresponding to the first two matrices of second signal autoencoder 
    int start=(A2.nsignals*A2.patchsize+1)*A2.nhidden0+(A2.nhidden0+1)*A2.nhidden1;

    //read and trash 'start' doubles
    double trash2;
    for(int i=0;i<start;++i)
      if(fscanf(readweights2,"%lf",&trash2)!=1){
	cout<<"problem reading data from file  "<<A2.backpropautoencodercoefficientsfile.c_str()<<endl;
	exit(-1);
      };     


    z=gsl_vector_fscanf (readweights2,&aux2.vector);

    if(z){

      cout<<"problem reading weights from file "<<A2.backpropautoencodercoefficientsfile.c_str()<<endl;

      exit(1);
    }

  //debug
  cout<<"after second autoencoder \n minimuweights norm is "<<gsl_blas_dnrm2 (minimumweights)<<endl;
 

  }

  //debug
  cout<<"minimuweights norm is "<<gsl_blas_dnrm2 (minimumweights)<<endl;

 
//-------------------------computer error rate before training------------

  {

    parametersfwdgsl parameters(numhidlayers,dimensions,npatches); 

    gsl_matrix_memcpy(&(parameters.reallayerdata[0].matrix),inputdata);

  
    double error_rate=just_compute_error_gsl_vislinear(minimumweights,&parameters,outputdata);


    cout<<"error rate before training is "<<error_rate*error_rate/npatches<<endl;


  }
 


  
  //-------------------------


  //have many test trials



  //gsl random number generator

  gsl_rng *r = gsl_rng_alloc(gsl_rng_taus2);
  unsigned long seed=time (NULL) * getpid();
  gsl_rng_set(r,seed);


  int permutation[npatches];
  for(int i=0;i<(int) npatches;++i)
    permutation[i]=i;


  //debug
  //numbatches=1;
  //numepochs=1;

  gsl_matrix *permutedinputdata=gsl_matrix_calloc(npatches,ninputs+1);

  gsl_matrix *permutedoutputdata=gsl_matrix_calloc(npatches,noutputs);

  gsl_vector_view auxrow;

  gsl_matrix_view auxinputview,auxoutputview;

  gsl_matrix_view real_permutedinputdata=gsl_matrix_submatrix(permutedinputdata,0,0,
							      npatches,ninputs);


  //permutedinputdata last column of ones
  gsl_vector_view v=gsl_matrix_column (permutedinputdata,ninputs);
  gsl_vector_add_constant (&v.vector, 1.0);



  for(int epoch=0;epoch<numepochs;++epoch){


    //randomly permute different segments of data

    gsl_ran_shuffle (r, permutation, npatches, sizeof(int));


    for(int i=0;i<(int) npatches;++i){

      auxrow=gsl_matrix_row (inputdata, permutation[i]);
     
      gsl_matrix_set_row (&real_permutedinputdata.matrix,i,&auxrow.vector);

      auxrow=gsl_matrix_row (outputdata, permutation[i]);
      
      gsl_matrix_set_row (permutedoutputdata,i,&auxrow.vector);     
    }

    for(int batch=0;batch<(int) numbatches;++batch){


      //inputbatchdata

      auxinputview=gsl_matrix_submatrix (permutedinputdata, batch*batchsize, 0, 
					 batchsize,ninputs+1);

      td.layerdata[0]=&auxinputview.matrix;



      //outputbatchdata

      auxoutputview=gsl_matrix_submatrix (permutedoutputdata, batch*batchsize, 0, 
      				  batchsize,noutputs);

      td.batchoutputdata=&auxoutputview.matrix;


      //initialing GSL-minimizer----------------------------

      const gsl_multimin_fdfminimizer_type *T;

      gsl_multimin_fdfminimizer *s;

      gsl_multimin_function_fdf my_func;

      my_func.n = td.tamtotal;

      my_func.f = compute_error_gsl_vislinear ;

      my_func.df = gradientgsl_vislinear  ;

      my_func.fdf = errorandgrad_vislinear ;

      my_func.params = &td;

      T = gsl_multimin_fdfminimizer_conjugate_pr;

      s = gsl_multimin_fdfminimizer_alloc (T, td.tamtotal);

      gsl_multimin_fdfminimizer_set (s, &my_func, minimumweights, 0.001, 1e-4);

      //-------


      //iterate minimaizer

      size_t iter = 0;

      size_t maxiterations=10;

      int status;

      do
	{
	  iter++;
	  status = gsl_multimin_fdfminimizer_iterate (s);
     
	  //test if everything is ok
	  if (status)
	    break;
     
	  status = gsl_multimin_test_gradient (s->gradient, 1e-3);

         
	}
      while (status == GSL_CONTINUE && iter < maxiterations);

      printf("epoch %4i batch %4i error rate  %e \r", epoch, batch,(s->f)*(s->f)/batchsize);
      fflush(stdout);


      gsl_vector_memcpy (minimumweights,s->x);

      gsl_multimin_fdfminimizer_free (s);

    }
    cout<<endl;
  }

  gsl_matrix_free(permutedinputdata);
  gsl_matrix_free(permutedoutputdata);

  gsl_rng_free (r);


  //save weights


  int matricesdimensions[]={5,nrows[0],ncols[0],nrows[1],ncols[1],nrows[2],ncols[2],
                                  nrows[3],ncols[3],nrows[4],ncols[4]};

  vector<int >  matdimensions(matricesdimensions,matricesdimensions+11);

  FILE * writeweights = fopen(f.c_str(), "w");

  savegslvectormatrix(writeweights, minimumweights,matdimensions); 

  fclose(writeweights);




//-------------------------computer error rate after training------------




  {

    parametersfwdgsl parameters(numhidlayers,dimensions,npatches); 

    gsl_matrix_memcpy(&(parameters.reallayerdata[0].matrix),inputdata);

    double error_rate=just_compute_error_gsl_vislinear(minimumweights,&parameters,outputdata);


    cout<<"error rate after training is "<<error_rate*error_rate/npatches<<endl;


  }
 


  
  //-------------------------




  //fwd data


  parametersfwdgsl tf(numhidlayers,dimensions,npatches); 

  gsl_matrix_memcpy(&tf.reallayerdata[0].matrix,inputdata);


  fwdgsl_vislinear(minimumweights,&tf,tf.fwd_data);

  


  //file for fwddata
  string fdata="../";
  fdata.append(cc);
  fdata.append("/fwd_trainingdata_mlp4layers_");
  fdata.append(signal1);
  fdata.append("_");
  fdata.append(signal2);
  fdata.append(".txt");


  writegslmatriz(fdata.c_str(),tf.fwd_data); 



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


