     /* backpropautoencodergsl.cpp - train the waits of an autoencoder using backprop algorithm
        Copyright (C) 2010  Rui Rodrigues <rapr@fct.unl.pt>
        This software is released under the terms of the GNU General
        Public License (http://www.gnu.org/copyleft/gpl.html).
     */


#include <stdio.h>
#include <string>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "gradgsl.h"
#include "netdimsandfilenames.h"
double initializeseed();
//in random_matrix.cpp

void checkfstream(ofstream& file_io,const char* filename);
//in ../geral/checkfstream.cpp

void checkfstream(ifstream& file_io,const char* filename);
//in ../geral/checkfstream.cpp

void read_datafile(ifstream&in,netdimsandfilenames& A);
//in netdimsandfilenames.cpp


void readgslvectormatrix(FILE* readfromfile, gsl_vector* v,vector<int>&sizes);
//in iosgslvectormatrix.cpp

//save a vector that contains several gslmatrices 
void savegslvectormatrix(FILE* writetofile, gsl_vector* v,vector<int>&sizes);
//in iosgslvectormatrix.cpp

void readgslmatriz(const char* filename,gsl_matrix*&m);
//in iogslvectormatrix.cpp

void writegslmatriz(const char* filename,gsl_matrix*m);
//in iosgslvectormatrix.cpp

int givesize_gslvector_infile(FILE* readfromfile);
//in iosgslvectormatrix.cpp


void useblacklist(gsl_matrix * &inputdata,const char*blacklistfile);
//in blacklist.cpp


//the following will be use to choose to load initial weights for rbmlogistic or backproprbm weights

const int num_matrices_autoencoder=4;

const string initial="initial";//from 'autoencoder_allcoefficientsgsl'

const string backprop="backprop";//backproprbm


struct  weightsfile{

  string filename;

  int num_matrices;

  int * whichmatrices;

  string matrixtype;//"initial" or "backprop"
  //informs if we take the weights from the file produced by 'prepareautoencoderforbackprop' or
  //by backproprbm
};


void fill_vector_weights(gsl_vector* weights,int weights_size,vector<weightsfile *>& weightsfilev,
			 int*nrows, int * ncols);
//down in this file



// -------------------------------------------------------------------



const string start="start";

const string cont="cont";

const string blacklist_use="useblacklist";

// ----------------------------------------------------------------------------------------
//-----------------------CONFIGURE-----------------------------------------------------

const int batchsize=504;

const int numepochs=50;

//-----------------------------------------------------------------------------------


int main(int argc, char ** argv){


  try{

if(argc<4){

    cout<<" must be called with argument start, startlogistic  cont and after signal name and folder name . Optionaly thereis an extra argument: useblacklist!"<<endl;

    exit(0);
  }


  int a=0;
  
  if(start.compare(argv[1])==0)
    a=1;

  if(cont.compare(argv[1])==0)
    a=2;

  if(a==0){

    cout<<"argument must be start or cont"<<endl;
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
  //gettimeofday(&start, NULL);
  //----------------------



 //get filename to extract dimensions and else--

  string signal=argv[2];
  string cc=argv[3];
  string b1=signal;
  b1.append(".txt");
  string folder="../";
  folder.append(cc);
  folder.append("/");
  string d1=folder;
  d1.append(b1);

  cout<<"reading configuration data from "<<d1.c_str()<<endl;

  netdimsandfilenames A;

  ifstream reading(d1.c_str());

  read_datafile(reading,A);

  checkfstream(reading,d1.c_str());

  reading.close();
  


  //-------------change in case of different number o layers---------------

  int patchsize=A.patchsize,
    nhidden0=A.nhidden0, nhidden1=A.nhidden1;

  int ninputs=A.nsignals*patchsize; 

  size_t numhidlayers=3;

  int dimensions[]={ninputs,nhidden0,nhidden1,nhidden0,ninputs};

  //--------------------------------------------------------------------


 //----------------------------------------------------
  //load training data


  string allpatches=folder;
  allpatches.append(signal);
  allpatches.append("_allpatches.txt");

  gsl_matrix * data;

  //cout<<"loading data from "<<A.patchdatafile.c_str()<<endl;

  string aim="aim"; 


  if(aim.compare(signal)==0)

    readgslmatriz(A.patchdatafile.c_str(),data);

  else

    readgslmatriz(allpatches.c_str(),data);


  if((int) data->size2!=ninputs){
    cout<<"inputdata is not compatible with ninputs!"<<endl;
    exit(1);
  }



  if(auxb==1){

    string blacklistfile=folder;
    blacklistfile.append(signal);
    blacklistfile.append("_blacklist.txt");    

    useblacklist(data,blacklistfile.c_str());
  }
 

  //----------------------------------------------------------


  //for backprop

  parametersgradgsl td(numhidlayers,dimensions,batchsize);  



  int npatches=(int) data->size1;

  int numbatches=(int) npatches/batchsize;
  

 

  //load weights 

  gsl_vector * minimumweights=gsl_vector_alloc (td.tamtotal);


  if(a==1){

  vector<int> sizes;

  FILE * readweights = fopen(A.autoencoderallcoeficientsgsl.c_str(), "r");

  readgslvectormatrix(readweights, minimumweights,sizes);


  }
  else if(a==2){

    vector<int> sizes;

    FILE * readweights = fopen(A.backpropautoencodercoefficientsfile.c_str(), "r");

    readgslvectormatrix(readweights, minimumweights,sizes);
  }
  else if(a==3){

    //const int num_matrices_autoencoder=4;
    //in the begining of this file


  int nrows[]={ninputs+1,A.nhidden0+1,A.nhidden1+1,A.nhidden0+1};
  int ncols[]={A.nhidden0,A.nhidden1,A.nhidden0,ninputs};


  //which weights matrix load from each file

  weightsfile wf1,wf2;

  //wf1
  wf1.filename= A.autoencoderallcoeficientsgsl;


  wf1.num_matrices=2;
  int matrices1[]={0,3};//these numbers refer to the matrices in the 
  //new autoencoder and in that one whose coefficients are in 'A.autoencoderallcoeficients'

  wf1.whichmatrices=matrices1;

  wf1.matrixtype="initial";


  //wf2
  wf2.filename=A.backprop_rbmlogisticweightsfile;

  wf2.num_matrices=2;

  int matrices2[]={1,2};

  wf2.whichmatrices=matrices2;

  wf2.matrixtype="backprop";

  vector<weightsfile *> weightsfvector(2);
  weightsfvector[0]=&wf1;
  weightsfvector[1]=&wf2;


  int weights_size=0;
  for(int i=0;i<num_matrices_autoencoder;++i)    
    weights_size+=nrows[i]*ncols[i];


  //gsl_vector *weights=gsl_vector_alloc(weights_size);

  fill_vector_weights(minimumweights,weights_size,weightsfvector,nrows, ncols);


  }
  //------------------

//-------------------------computer error rate before training------------

  {

    parametersfwdgsl parameters(numhidlayers,dimensions,npatches); 

    gsl_matrix_memcpy(&(parameters.reallayerdata[0].matrix),data);

    
    double error_rate=just_compute_error_gsl_vislinear(minimumweights,&parameters,data);

    cout<<"error rate before training is "<<error_rate*error_rate/npatches<<endl;

  }
 


  
  //-------------------------


  
  //---------------------------------------------------------------------


  //have many test trials


  //gsl random number generator

  gsl_rng *r = gsl_rng_alloc(gsl_rng_taus2);
  int long seed=time (NULL) * getpid();
  gsl_rng_set(r,seed);



  int permutation[npatches];

  for(int i=0;i<npatches;++i)
    permutation[i]=i;

  gsl_matrix *permutedinputdata=gsl_matrix_calloc(npatches,ninputs+1);

  gsl_vector_view auxrow;

  gsl_matrix_view auxinputview,auxoutputview;


  //permutedinputdata last column of ones
  gsl_vector_view v=gsl_matrix_column (permutedinputdata,ninputs);
  gsl_vector_add_constant (&v.vector, 1.0);


  gsl_matrix_view real_permutedinputdata=gsl_matrix_submatrix(permutedinputdata,0,0,
							      npatches,ninputs);




  //debug
  //numbatches=1;  



  for(int epoch=0;epoch<numepochs;++epoch){


    //randomly permute different segments of data
    gsl_ran_shuffle (r, permutation, npatches, sizeof(int));


    for(int i=0;i<npatches;++i){

      auxrow=gsl_matrix_row (data, permutation[i]);
     
      gsl_matrix_set_row (&real_permutedinputdata.matrix,i,&auxrow.vector);

    }


    for(int batch=0;batch<numbatches;++batch){


      //Batchdata

      auxinputview=gsl_matrix_submatrix (permutedinputdata, batch*batchsize, 0, 
					 batchsize,ninputs+1);

      td.layerdata[0]=&auxinputview.matrix;

      auxoutputview=gsl_matrix_submatrix (permutedinputdata, batch*batchsize, 0, 
      				  batchsize,ninputs);

      td.batchoutputdata=&auxoutputview.matrix;
 


      //initialing GSL-minimizer----------------------------

      const gsl_multimin_fdfminimizer_type *T;

      gsl_multimin_fdfminimizer *s;

      gsl_multimin_function_fdf my_func;

      my_func.n = td.tamtotal;

      my_func.f = compute_error_gsl_vislinear ;

      my_func.df = gradientgsl_vislinear ;

      my_func.fdf = errorandgrad_vislinear;

      my_func.params = &td;

      T = gsl_multimin_fdfminimizer_conjugate_fr;

      s = gsl_multimin_fdfminimizer_alloc (T, td.tamtotal);

      gsl_multimin_fdfminimizer_set (s, &my_func, minimumweights, 0.01, 1e-4);

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

	  //printf ("%5d %10.5f\n", iter,s->f);
         
	}
      while (status == GSL_CONTINUE && iter < maxiterations);

      printf("epoch %4i batch %4i error rate by patch %e \r", epoch, batch,(s->f)*(s->f)/batchsize);
      fflush(stdout);


      gsl_vector_memcpy (minimumweights,s->x);

      gsl_multimin_fdfminimizer_free (s);

    }
    //cout<<endl;
  }

  gsl_matrix_free(permutedinputdata);

  gsl_rng_free (r);

  //save weights


  int matricesdimensions[]={4,dimensions[0]+1,dimensions[1],dimensions[1]+1,dimensions[2],dimensions[2]+1,dimensions[3],
                                  dimensions[3]+1,dimensions[4]};

  vector<int>  matdimensions(matricesdimensions,matricesdimensions+9);

  FILE * writeweights = fopen(A.backpropautoencodercoefficientsfile.c_str(), "w");

  savegslvectormatrix(writeweights, minimumweights,matdimensions); 

  fclose(writeweights);



  
  //-------------------computer error rate after training------------

  {
    //it could/should be like we did before training

    parametersgradgsl parameters(numhidlayers,dimensions,npatches); 

    parameters.batchoutputdata=data;
  
    gsl_matrix * inputdata=gsl_matrix_calloc(npatches,ninputs+1);

    v=gsl_matrix_column (inputdata,ninputs);
    gsl_vector_add_constant (&v.vector, 1.0);

    gsl_matrix_view realdata0=gsl_matrix_submatrix(inputdata,0,0,npatches,ninputs);

    gsl_matrix_memcpy(&realdata0.matrix,data);

    parameters.layerdata[0]=inputdata;

    
    double error_rate=compute_error_gsl_vislinear(minimumweights,&parameters);

    cout<<"error rate after training is "<<error_rate*error_rate/npatches<<endl;

    gsl_matrix_free(inputdata);

  }
   
  //-------------------------

  


  gettimeofday(&start, NULL);
  

  //fwd data

  parametersfwdgsl tf(numhidlayers,dimensions,npatches); 

  gsl_matrix_memcpy(&tf.reallayerdata[0].matrix,data);



  //for(int cycle=0;cycle<5;++cycle)

  fwdgsl_vislinear(minimumweights,&tf,tf.fwd_data);




  
  gsl_vector_free(minimumweights);  


  writegslmatriz(A.fwdautoencoderdata_afterbackprop_file.c_str(),
		 tf.fwd_data);




  gsl_matrix_free(data); 

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







void fill_vector_weights(gsl_vector* weights,int weights_size,vector<weightsfile *>& weightsfilev,
			 int*nrows, int * ncols){

  int numfiles=weightsfilev.size();

  for(int i=0;i<numfiles;++i){

    if(weightsfilev[i]->matrixtype.compare(initial)==0){


      //load coefficients from this file into vector gslvecaux

      FILE * readweights = fopen(weightsfilev[i]->filename.c_str(), "r");

      int tam=givesize_gslvector_infile(readweights);

      rewind (readweights);

      gsl_vector*gslvecaux=gsl_vector_calloc(tam);

      vector<int> sizes(num_matrices_autoencoder);

      readgslvectormatrix(readweights, gslvecaux, sizes);


      //insert those we wnt into vector weights

      for(int j=0;j<weightsfilev[i]->num_matrices;++j){
      
	int matrixorder=weightsfilev[i]->whichmatrices[j];

	int begin=0;
	for(int k=0;k<matrixorder;++k){
	  begin+=nrows[k]*ncols[k];
	  
	  //debug
	  if(sizes[2*k+1]*sizes[2*k+2]!=nrows[k]*ncols[k]){
	    cout<<"sizes[2*k+1]*sizes[2*k+2]!=nrows[k]*ncols[k]"<<endl;
	    char a;cin>>a;
	  }
	}

	for(int n=0;n<nrows[matrixorder]*ncols[matrixorder];++n)
	  gsl_vector_set(weights,begin+n,gsl_vector_get(gslvecaux,begin+n));
      }
      	gsl_vector_free(gslvecaux);
    }

    else if(weightsfilev[i]->matrixtype.compare(backprop)==0){

      //load coefficients into vector gslvecaux

     FILE * readweights = fopen(weightsfilev[i]->filename.c_str(), "r");

      int tam=givesize_gslvector_infile(readweights);

      rewind (readweights);

      gsl_vector*gslvecaux=gsl_vector_calloc(tam);

      vector<int> sizes(weightsfilev[i]->num_matrices);

      readgslvectormatrix(readweights, gslvecaux, sizes);


      //insert those coefficients in the appropriate place in vector weights

      for(int j=0;j<weightsfilev[i]->num_matrices;++j){

	int matrixorder=weightsfilev[i]->whichmatrices[j];

	int beginweights=0;
	for(int k=0;k<matrixorder;++k)
	  beginweights+=nrows[k]*ncols[k];

	int beginaux=0;
	for(int z=0;z<j;++z){

	  int order=weightsfilev[i]->whichmatrices[z];

	  beginaux+=nrows[order]*ncols[order];
	}


	for(int n=0;n<nrows[matrixorder]*ncols[matrixorder];++n)
	  gsl_vector_set(weights,beginweights+n,gsl_vector_get(gslvecaux,beginaux+n));

      }
      gsl_vector_free(gslvecaux);

    }


  }

}
