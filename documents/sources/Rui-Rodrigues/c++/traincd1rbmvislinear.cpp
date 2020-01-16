
     /* trabcd1rbmvislinear.cpp - train rbmvislinear in cd-1 way (random zero one output in the hidden layer)
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

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_blas.h>
#include "netdimsandfilenames.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_permutation.h>

#ifdef _OPENMP
#include <omp.h>
#endif



void checkfstream(ofstream& file_io,const char* filename);
//in ../checkfstream.cpp

void checkfstream(ifstream& file_io,const char* filename);
//in ../checkfstream.cpp


void read_datafile(ifstream&in,netdimsandfilenames& A);
//in netdimsandfilenames.cpp


void logistic (gsl_matrix * m,size_t nrows,size_t ncols);
//down in this file

void writegslmatriz(const char* filename,gsl_matrix*m);
void readgslmatriz(const char* filename,gsl_matrix*&m);
void readgslvector(const char* filename,gsl_vector*m);
void writegslvector(const char* filename,gsl_vector*m);
//in iogslvectormatrix.cpp

void useblacklist(gsl_matrix * &inputdata,const char*blacklistfile);
//in blacklist.cpp

// ----------------------------------------------------------------------------------------
//-----------------------CONFIGURE-----------------------------------------------------

const size_t batchsize=500;

const size_t numepochs=100;

const double epsilonweights=0.01;

const double epsilonbias=0.01;

const double momentum=0.2;

const double weightscost=0.0002;


//-----------------------------------------------------------------------------------





class netrbmvislinear{

public:

  netrbmvislinear(unsigned ninputs,unsigned nhidden, unsigned numcases,
		  unsigned Maxnumthreads,unsigned Blocksize, unsigned Npatches,
		  gsl_vector *Vectorweightsandbias);

  ~netrbmvislinear();


  //data

  gsl_matrix  *permutedtdata, *hiddenlayer, * batchimagedata, *posprobs, *negprobs, *hiddenactivation,
    *posprods, *negprods, *deltapureweights;

  gsl_vector *vectorweightsandbias,*deltavisbias, *deltahidbias, *sumbatchtdata, *sumbatchimagedata, *sumposprobs, *sumnegprobs;

  gsl_matrix_view  batchtdata, *blocktdata, *blockposprobs,  *blocknegprobs, *blockimagedata, *blockhiddenactivation,pureweights;

  gsl_vector_view visbias, hidbias;

  unsigned ninputs, nhidden, numcases, maxnumthreads, *sizesblocks, blocksize, npatches;

};




netrbmvislinear::netrbmvislinear(unsigned Ninputs,unsigned Nhidden, unsigned Numcases, 
				 unsigned Maxnumthreads, unsigned Blocksize, unsigned Npatches,
				 gsl_vector *Vectorweightsandbias){


  ninputs=Ninputs;

  nhidden=Nhidden;

  numcases=Numcases;

  maxnumthreads=Maxnumthreads;

  blocksize=Blocksize;

  npatches=Npatches;

  vectorweightsandbias=Vectorweightsandbias;//just copy the pointer


  //numcases is batchsize or npatches
 

  
  permutedtdata=gsl_matrix_alloc(npatches,ninputs);

  posprobs=gsl_matrix_calloc(numcases,nhidden);

  negprobs=gsl_matrix_calloc(numcases,nhidden);

  hiddenactivation=gsl_matrix_calloc(numcases,nhidden);

  batchimagedata=gsl_matrix_calloc(numcases,ninputs);   

  posprods=gsl_matrix_calloc(ninputs,nhidden);

  negprods=gsl_matrix_calloc(ninputs,nhidden);

  deltapureweights=gsl_matrix_calloc(ninputs,nhidden);


  deltavisbias=gsl_vector_alloc(ninputs);

  deltahidbias=gsl_vector_alloc(nhidden);
  


  sizesblocks=new unsigned[maxnumthreads];

  for(unsigned i=0;i<maxnumthreads-1;++i)
    sizesblocks[i]=blocksize;

  sizesblocks[maxnumthreads-1]=numcases-(maxnumthreads-1)*blocksize;


  blocktdata=new gsl_matrix_view[maxnumthreads];

  blockimagedata=new gsl_matrix_view[maxnumthreads];

    
  for(unsigned thread=0;thread<maxnumthreads;++thread)
       
    blockimagedata[thread]= gsl_matrix_submatrix(batchimagedata,thread*blocksize,0,
						   sizesblocks[thread],ninputs);

  blockposprobs=new gsl_matrix_view[maxnumthreads];

    
  for(unsigned thread=0;thread<maxnumthreads;++thread)
       
    blockposprobs[thread]= gsl_matrix_submatrix(posprobs,thread*blocksize,0,
						sizesblocks[thread],nhidden);

  //blockposprobs[maxnumthreads-1]=gsl_matrix_submatrix(posprobs,thread*blocksize,0,
  //						sizesblocks[],nhidden);


  blocknegprobs=new gsl_matrix_view[maxnumthreads];

    
  for(unsigned thread=0;thread<maxnumthreads;++thread)
       
    blocknegprobs[thread]= gsl_matrix_submatrix(negprobs,thread*blocksize,0,
						sizesblocks[thread],nhidden);


  blockhiddenactivation=new gsl_matrix_view[maxnumthreads];

    
  for(unsigned thread=0;thread<maxnumthreads;++thread)
       
    blockhiddenactivation[thread]= gsl_matrix_submatrix(hiddenactivation,thread*blocksize,0,
						sizesblocks[thread],nhidden);



  sumbatchtdata=gsl_vector_calloc(ninputs);

  
  sumbatchimagedata=gsl_vector_calloc(ninputs);
 
  sumposprobs=gsl_vector_calloc(nhidden);

  sumnegprobs=gsl_vector_calloc(nhidden);


  //weights and bias

  pureweights=gsl_matrix_view_vector(vectorweightsandbias,ninputs, nhidden);

  visbias=gsl_vector_subvector(vectorweightsandbias, ninputs*nhidden,ninputs);  

  hidbias=gsl_vector_subvector(vectorweightsandbias, ninputs*nhidden+ninputs,
			       nhidden);  
}




netrbmvislinear::~netrbmvislinear(){


  gsl_matrix_free(permutedtdata);

  gsl_matrix_free(posprobs);

  gsl_matrix_free(negprobs);

  gsl_matrix_free(hiddenactivation);

  gsl_matrix_free(batchimagedata);

  gsl_matrix_free(posprods);

  gsl_matrix_free(negprods);


  gsl_vector_free(sumbatchtdata);

  gsl_vector_free(sumbatchimagedata);

  gsl_vector_free(sumposprobs);

  gsl_vector_free(sumnegprobs);
 
  gsl_matrix_free(deltapureweights);

  gsl_vector_free(deltavisbias);

  gsl_vector_free(deltahidbias);


  delete [] sizesblocks;

  delete [] blocktdata;

  //delete [] blockhiddendata;

  delete [] blockimagedata;

  delete [] blockposprobs;

  delete [] blocknegprobs;
}



double compute_error_rbmvislinear(gsl_vector * vectorweightsandbias, gsl_matrix * data,
				  unsigned ninputs,unsigned nhidden);
//down in this file


void cd1trainrbmvislinear(unsigned numepochs,gsl_vector *vectorweightsandbias,
			  gsl_matrix *tdata,unsigned ninputs,unsigned nhidden);
//down in this file


const string start="start";

const string cont="cont";


const string blacklist_use="useblacklist";



int main(int argc, char ** argv){


  try{

    if(argc<4){

    cout<<" must be called with argument start or cont, after signal1 and  folder name. Optionaly thereis an extra argument: useblacklist !"<<endl;

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


  //signal

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

  netdimsandfilenames A1;

  ifstream reading1(d1.c_str());

  read_datafile(reading1,A1);

  checkfstream(reading1,d1.c_str());

  reading1.close();
  

  unsigned ninputs=A1.nsignals*A1.patchsize;

  unsigned nhidden=A1.nhidden0;


  //load training data ;

  string allpatches=folder;
  allpatches.append(signal);
  allpatches.append("_allpatches.txt");


  gsl_matrix * data;


  string aim="aim"; 


  if(aim.compare(argv[2])==0)

    readgslmatriz(A1.patchdatafile.c_str(),data);

  else

    readgslmatriz(allpatches.c_str(),data);



  if(data->size2!=ninputs){
    cout<<"inputdata is not compatible with ninputs!"<<endl;
    exit(1);
  }



  if(auxb==1){

    string blacklistfile=folder;
    blacklistfile.append(signal);
    blacklistfile.append("_blacklist.txt");    

    useblacklist(data,blacklistfile.c_str());
  }

  size_t npatches=data->size1;


  //gsl random number generator

  gsl_rng *r = gsl_rng_alloc(gsl_rng_taus2);

  unsigned long seed=time (NULL) * getpid();

  gsl_rng_set(r,seed);



  //load weights (first matrix with pure weights then visible bias finally 
  //hidden bias 

  gsl_vector * vectorweightsandbias=gsl_vector_calloc (ninputs*nhidden+ninputs+nhidden);



  if(a==1)

  //random weights

    for(unsigned i=0;i<vectorweightsandbias->size;++i)

      gsl_vector_set(vectorweightsandbias,i,gsl_ran_gaussian(r,0.01));

 else if(a==2)

  //load weights from file

  readgslvector(A1.netvislinearweights.c_str(),vectorweightsandbias);


 //computer error rate before training

  double error=compute_error_rbmvislinear(vectorweightsandbias, data,ninputs,nhidden);

  cout<<"error rate by patch before training is "<<error*error/npatches<<endl;


  
  //train

  //unsigned numepochs=25;

  cd1trainrbmvislinear(numepochs,vectorweightsandbias,data,ninputs,nhidden);



  //save weights

  writegslvector(A1.netvislinearweights.c_str(),vectorweightsandbias);


   //computer error rate after training

  error=compute_error_rbmvislinear(vectorweightsandbias, data,ninputs,nhidden);

  cout<<"error rate by patch after training is "<<error*error/npatches<<endl;



  //time

  gettimeofday(&end, NULL);

  int seconds  = end.tv_sec  - start.tv_sec;
  int useconds = end.tv_usec - start.tv_usec;

  int mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;

  cout<<"Elapsed time: "<<mtime<<" milliseconds\n"<<endl;




  //free

  gsl_matrix_free(data);

  gsl_rng_free(r);

  gsl_vector_free(vectorweightsandbias);
  }

  catch(int i){
    cout<<"exception "<<i<<endl;
    exit(1);
  }


  return 0;

  }









  //-----------------------------------------------------------------------------------



  double compute_error_rbmvislinear(gsl_vector * vectorweightsandbias, gsl_matrix * data,unsigned ninputs,unsigned nhidden){

#ifdef _OPENMP
  int maxnumthreads = omp_get_max_threads();
#else /* _OPENMP */
  int maxnumthreads = 4;
#endif /* _OPENMP */

    unsigned npatches=data->size1;

    const unsigned blocksize=npatches/maxnumthreads;


    gsl_matrix*hiddendata=gsl_matrix_calloc(npatches,nhidden);

    gsl_vector*vectorimagedata=gsl_vector_calloc(npatches*ninputs);   

    gsl_matrix_view imagedata=gsl_matrix_view_vector(vectorimagedata,npatches,ninputs);


    //block views to previous matrices



    unsigned *sizesblocks=new unsigned[maxnumthreads];


    for(int i=0;i<maxnumthreads-1;++i)
      sizesblocks[i]=blocksize;

    sizesblocks[maxnumthreads-1]=npatches-(maxnumthreads-1)*blocksize;
    
    


    gsl_matrix_view * blockinputdata=new gsl_matrix_view[maxnumthreads];

    
    for(int thread=0;thread<maxnumthreads;++thread)
       
      blockinputdata[thread]= gsl_matrix_submatrix(data,thread*blocksize,0,
						   sizesblocks[thread],ninputs);


    gsl_matrix_view * blockhiddendata=new gsl_matrix_view[maxnumthreads];

    
    for(int thread=0;thread<maxnumthreads;++thread)
       
      blockhiddendata[thread]= gsl_matrix_submatrix(hiddendata,thread*blocksize,0,
						   sizesblocks[thread],nhidden);
   
    
    gsl_matrix_view * blockimagedata=new gsl_matrix_view[maxnumthreads];

    
    for(int thread=0;thread<maxnumthreads;++thread)
       
      blockimagedata[thread]= gsl_matrix_submatrix(&imagedata.matrix,thread*blocksize,0,
						   sizesblocks[thread],ninputs);



    //weights and bias

    gsl_matrix_view pureweights=gsl_matrix_view_vector(vectorweightsandbias,ninputs, nhidden);

    gsl_vector_view visbias=gsl_vector_subvector(vectorweightsandbias, ninputs*nhidden,ninputs);  

    gsl_vector_view hidbias=gsl_vector_subvector(vectorweightsandbias, ninputs*nhidden+ninputs,
					       nhidden); 




#pragma omp parallel for
    for(int thread=0;thread<maxnumthreads;++thread){


      //add hidden bias
      for(unsigned i=0;i<sizesblocks[thread];++i)

	gsl_matrix_set_row (&blockhiddendata[thread].matrix,i,&hidbias.vector);


      //multiply weights
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
		 1.0, &blockinputdata[thread].matrix, &pureweights.matrix,
		 1.0, &blockhiddendata[thread].matrix);


	logistic (&blockhiddendata[thread].matrix,sizesblocks[thread],nhidden);

	
      //add visible bias
      for(unsigned i=0;i<sizesblocks[thread];++i)

	gsl_matrix_set_row (&blockimagedata[thread].matrix,i,&visbias.vector);
	

      //multiply weights
        gsl_blas_dgemm(CblasNoTrans, CblasTrans,
		 1.0, &blockhiddendata[thread].matrix, &pureweights.matrix,
		 1.0, &blockimagedata[thread].matrix);


	//activation function is identity

	//-error matrix

	gsl_matrix_sub (&blockimagedata[thread].matrix,&blockinputdata[thread].matrix);
    }

    double error=gsl_blas_dnrm2 (vectorimagedata);

    //free

    gsl_matrix_free(hiddendata);

    gsl_vector_free(vectorimagedata);   

    delete [] blockinputdata;

    delete [] blockhiddendata;

    delete [] blockimagedata;


    return error;

  }





//--------------------------------cd1train-----------------------------------------------


//aux

double sum(gsl_vector*vvector){


  double adition=0.0;


  for(unsigned i=0;i<vvector->size;++i)

    adition+=gsl_vector_get(vvector,i);

  return adition;

}




//this part is not parallel
void updateweightsandbias(netrbmvislinear*ptnet){



  //pure weights

  gsl_matrix_sub (ptnet->posprods,ptnet->negprods);



  gsl_matrix_scale(ptnet->posprods,epsilonweights/ptnet->numcases);

  gsl_matrix_scale(ptnet->deltapureweights, momentum);

  gsl_matrix_add(ptnet->deltapureweights,ptnet->posprods);

  gsl_matrix_scale(&ptnet->pureweights.matrix ,1.0-epsilonweights*weightscost);

  gsl_matrix_add( &ptnet->pureweights.matrix, ptnet->deltapureweights);




  gsl_vector_view aux;


  //visbias
  
  for(unsigned i=0;i<ptnet->ninputs;++i){

    aux=gsl_matrix_column(&ptnet->batchtdata.matrix,i);

    gsl_vector_set(ptnet->sumbatchtdata,i,sum(&aux.vector));

  }


  for(unsigned i=0;i<ptnet->ninputs;++i){

    aux=gsl_matrix_column(ptnet->batchimagedata,i);

    gsl_vector_set(ptnet->sumbatchimagedata,i,sum(&aux.vector));

  }


  gsl_vector_sub(ptnet->sumbatchtdata,ptnet->sumbatchimagedata);


  gsl_vector_scale(ptnet->sumbatchtdata,epsilonweights/ptnet->numcases);


  gsl_vector_scale(ptnet->deltavisbias,momentum);


  gsl_vector_add(ptnet->deltavisbias,ptnet->sumbatchtdata);

  gsl_vector_scale(&ptnet->visbias.vector, 1-epsilonweights*weightscost);

  gsl_vector_add(&ptnet->visbias.vector,ptnet->deltavisbias);



  //hidbias


  for(unsigned i=0;i<ptnet->nhidden;++i){

    aux=gsl_matrix_column(ptnet->posprobs,i);

    gsl_vector_set(ptnet->sumposprobs,i,sum(&aux.vector));

  }



  for(unsigned i=0;i<ptnet->nhidden;++i){

    aux=gsl_matrix_column(ptnet->negprobs,i);

    gsl_vector_set(ptnet->sumnegprobs,i,sum(&aux.vector));

  }


  gsl_vector_sub(ptnet->sumposprobs,ptnet->sumnegprobs);


  gsl_vector_scale(ptnet->sumposprobs,epsilonweights/ptnet->numcases);


  gsl_vector_scale(ptnet->deltahidbias,momentum);


  gsl_vector_add(ptnet->deltahidbias,ptnet->sumposprobs);


  gsl_vector_scale(&ptnet->hidbias.vector, 1-epsilonweights*weightscost);


  gsl_vector_add(&ptnet->hidbias.vector,ptnet->deltahidbias);

}



void cd1trainrbmvislinear(unsigned numepochs,gsl_vector *vectorweightsandbias,
			  gsl_matrix *tdata,unsigned ninputs,unsigned nhidden){



#ifdef _OPENMP
  int maxnumthreads = omp_get_max_threads();
#else /* _OPENMP */
  int maxnumthreads = 4;
#endif /* _OPENMP */

  
  unsigned npatches=tdata->size1;

  const unsigned blocksize=batchsize/maxnumthreads;

  unsigned numbatches=npatches/batchsize;

  netrbmvislinear  net(ninputs, nhidden, batchsize,
		       maxnumthreads, blocksize, npatches,vectorweightsandbias);



  //gsl random number generator

  gsl_rng *r = gsl_rng_alloc(gsl_rng_taus2);

  unsigned long seed=time (NULL) * getpid();

  gsl_rng_set(r,seed);


  unsigned permutation[npatches];

  for(unsigned i=0;i<npatches;++i)
    permutation[i]=i;

  gsl_vector_view auxrow;



  for(unsigned epoch=0;epoch<numepochs;++epoch){


    //randomly permute different segments of data------------

    gsl_ran_shuffle (r, permutation, npatches, sizeof(unsigned));

    for(unsigned i=0;i<npatches;++i){


      auxrow=gsl_matrix_row (tdata, permutation[i]);

      
      gsl_matrix_set_row (net.permutedtdata,i,&auxrow.vector);

    }


    //debug
    //numbatches=3;

   for(unsigned batch=0;batch<numbatches;++batch){


     //initialize data

     net.batchtdata=gsl_matrix_submatrix(net.permutedtdata,batch*batchsize,0,batchsize,ninputs);

     for(int thread=0;thread<maxnumthreads;++thread)
       
       net.blocktdata[thread]= gsl_matrix_submatrix(&net.batchtdata.matrix,thread*blocksize,0,
						 net.sizesblocks[thread],ninputs);


     gsl_matrix_set_zero(net.posprobs);
     
     gsl_matrix_set_zero(net.negprobs);

     gsl_matrix_set_zero(net.hiddenactivation);

     gsl_matrix_set_zero(net.batchimagedata);

     gsl_matrix_set_zero(net.deltapureweights);

     gsl_vector_set_zero(net.sumbatchtdata);

     gsl_vector_set_zero(net.sumbatchimagedata);

     gsl_vector_set_zero(net.sumposprobs);

     gsl_vector_set_zero(net.sumnegprobs);

     gsl_vector_set_zero(net.deltavisbias);

     gsl_vector_set_zero(net.deltahidbias);

  

#pragma omp parallel for 
     for(int thread=0;thread<maxnumthreads;++thread){


      //add hidden bias
      for(unsigned i=0;i<net.sizesblocks[thread];++i)
	gsl_matrix_set_row(&net.blockposprobs[thread].matrix,i,&net.hidbias.vector);

      //multiply weights
      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
		     1.0, &net.blocktdata[thread].matrix, &net.pureweights.matrix,
		     1.0, &net.blockposprobs[thread].matrix);


      logistic (&net.blockposprobs[thread].matrix,net.sizesblocks[thread],nhidden);


      gsl_matrix_memcpy ( &net.blockhiddenactivation[thread].matrix ,&net.blockposprobs[thread].matrix);


      //generate uniform random data and compare with blockhiddenactivation
      //if the blockhiddenactivation is bigger it gets one otherwise zero


      for(unsigned i=0;i<net.sizesblocks[thread];++i)
	for(unsigned j=0;j<nhidden;++j)
	  if (gsl_matrix_get(&net.blockhiddenactivation[thread].matrix ,i,j)>gsl_rng_uniform (r))	  
	    gsl_matrix_set(&net.blockhiddenactivation[thread].matrix ,i,j,1.0);
	  else
	    gsl_matrix_set(&net.blockhiddenactivation[thread].matrix ,i,j,0.0);


      //add visble bias

      for(unsigned i=0;i<net.sizesblocks[thread];++i)
	gsl_matrix_set_row(&net.blockimagedata[thread].matrix,i,&net.visbias.vector);


	//multiply weights
        gsl_blas_dgemm(CblasNoTrans, CblasTrans,
		       1.0, &net.blockhiddenactivation[thread].matrix,
		       &(net.pureweights).matrix,
		       1.0, &net.blockimagedata[thread].matrix);

	//activation function is identity


	//add hidden bias
	for(unsigned i=0;i<net.sizesblocks[thread];++i)
	  gsl_matrix_set_row(&net.blocknegprobs[thread].matrix,i,&net.hidbias.vector);


	//multiply weights
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
		       1.0, &net.blockimagedata[thread].matrix, &net.pureweights.matrix,
		       1.0, &net.blocknegprobs[thread].matrix);


	logistic (&net.blocknegprobs[thread].matrix,net.sizesblocks[thread],nhidden);


     }

  

     //posprods
     gsl_blas_dgemm(CblasTrans, CblasNoTrans,
		    1.0, &net.batchtdata.matrix, net.posprobs,
		    0.0, net.posprods);

     //negprods
     gsl_blas_dgemm(CblasTrans, CblasNoTrans,
		    1.0, net.batchimagedata, net.negprobs,
		    0.0, net.negprods);

     updateweightsandbias(&net);

     gsl_matrix_sub(net.batchimagedata,&net.batchtdata.matrix);

     gsl_vector_view aauuxx=gsl_vector_view_array(net.batchimagedata->data,batchsize*net.ninputs);

     double error=gsl_blas_dnrm2 (&aauuxx.vector);

     printf("epoch %4i batch %4i  error rate by patche is %e \r", epoch, batch,error*error/batchsize);
     fflush(stdout); 
   }


  }

}


void logistic (gsl_matrix * m,size_t nrows,size_t ncols){

  for(unsigned i=0;i<nrows;++i)
    for(unsigned j=0;j<ncols;++j)
      gsl_matrix_set(m,i,j,1/(1+exp((-1)*gsl_matrix_get(m,i,j))));

}
