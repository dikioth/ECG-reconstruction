
     /* trainmeanfieldrbmlogistic.cpp - train rbmlogistic as a deterministic network
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

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_permutation.h>

#ifdef _OPENMP
#include <omp.h>
#endif



void checkfstream(ofstream& file_io,const char* filename);
//in checkfstream.cpp

void checkfstream(ifstream& file_io,const char* filename);
//in checkfstream.cpp


void read_datafile(ifstream&in,netdimsandfilenames& A);
//in netdimsandfilenames.cpp


void logistic (gsl_matrix * m);
//in down in this file

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

const double epsilonweights=0.1;

const double epsilonbias=0.1;

const double momentum=0.2;

const double weightscost=0.002;


//-----------------------------------------------------------------------------------






double compute_error_rbmlogistic(netrbm&, gsl_matrix * data);
//down in this file


void meanfieldtrainrbmlogistic(unsigned numepochs,gsl_vector *vectorweightsandbias,
			  gsl_matrix *tdata,unsigned ninputs,unsigned nhidden);
//down in this file


const string start="start";

const string cont="cont";

const string blacklist_use="useblacklist";



int main(int argc, char ** argv){


  try{

if(argc<3){

    cout<<" must be called with argument  signal1 and after folder name. Optionaly thereis an extra argument: useblacklist !"<<endl;

    exit(0);
  }


  int auxb=0;


  if(argc==4)
    if(blacklist_use.compare(argv[3])==0)
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
  

  unsigned ninputs=A1.nhidden0;

  unsigned nhidden=A1.nhidden1;


  //load training data (patchdatafile);

  gsl_matrix * data;
 
  readgslmatriz(A1.rbmvislinearhiddendatafile.c_str(),data);

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



  //load weights from file

  readgslvector(A1.netlogisticweights.c_str(),vectorweightsandbias);


#ifdef _OPENMP
  int maxnumthreads = omp_get_max_threads();
#else /* _OPENMP */
  int maxnumthreads = 4;
#endif /* _OPENMP */


  unsigned blocksize=npatches/maxnumthreads;//just fo computing error


 //computer error rate before training

  {

   

    netrbm net1(ninputs,nhidden, npatches, maxnumthreads, blocksize, npatches,
	   vectorweightsandbias);

  double error=compute_error_rbmlogistic(net1,data);

  cout<<"error rate by patch before training is "<<error*error/npatches<<endl;

  }
  
  //train

 

  meanfieldtrainrbmlogistic(numepochs,vectorweightsandbias,data,ninputs,nhidden);

 


  //save weights

  writegslvector(A1.netlogisticweights.c_str(),vectorweightsandbias);


  //computer error rate after training



   netrbm net2(ninputs,nhidden, npatches, maxnumthreads, blocksize, npatches,
	   vectorweightsandbias);

  double error=compute_error_rbmlogistic(net2, data);

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



double compute_error_rbmlogistic(netrbm& net, gsl_matrix * data){


  gsl_matrix_memcpy(net.permutedtdata,data);


  //initialize data

  net.batchtdata=gsl_matrix_submatrix(net.permutedtdata,0,0,net.npatches,net.ninputs);

  for(unsigned thread=0;thread<net.maxnumthreads;++thread)
       
    net.blocktdata[thread]= gsl_matrix_submatrix(&net.batchtdata.matrix,thread*net.blocksize,0,
						 net.sizesblocks[thread],net.ninputs);

  gsl_matrix_set_zero(net.posprobs);

  gsl_matrix_set_zero(net.hiddenactivation);

  gsl_matrix_set_zero(net.batchimagedata);
  


#pragma omp parallel for
  for(int thread=0;thread<(int) net.maxnumthreads;++thread){


    //add hidden bias
    for(unsigned i=0;i<net.sizesblocks[thread];++i)

      gsl_matrix_set_row (&net.blockposprobs[thread].matrix,i,&net.hidbias.vector);


    //multiply weights
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
		   1.0, &net.blocktdata[thread].matrix, &net.pureweights.matrix,
		   1.0, &net.blockposprobs[thread].matrix);


    logistic (&net.blockposprobs[thread].matrix);

	
    //add visible bias
    for(unsigned i=0;i<net.sizesblocks[thread];++i)

      gsl_matrix_set_row (&net.blockimagedata[thread].matrix,i,&net.visbias.vector);
	

    //multiply weights
    gsl_blas_dgemm(CblasNoTrans, CblasTrans,
		   1.0, &net.blockposprobs[thread].matrix, &net.pureweights.matrix,
		   1.0, &net.blockimagedata[thread].matrix);


    //activation function is identity

    logistic (&net.blockimagedata[thread].matrix);


    //-error matrix

    gsl_matrix_sub (&net.blockimagedata[thread].matrix,&net.blocktdata[thread].matrix);
  }

  gsl_vector_view vectorimagedata=gsl_vector_view_array (net.batchimagedata->data,
							 net.npatches*net.ninputs);

  double error=gsl_blas_dnrm2 (&vectorimagedata.vector);

   
  return error;

}





//--------------------------------meanfieldtrain-----------------------------------------------

void logistic (gsl_matrix * m){

  unsigned nrows=m->size1, ncols=m->size2;

  for(unsigned i=0;i<nrows;++i)
    for(unsigned j=0;j<ncols;++j)
      gsl_matrix_set(m,i,j,1/(1+exp((-1)*gsl_matrix_get(m,i,j))));

}

//aux

double sum(gsl_vector*vvector){


  double adition=0.0;


  for(unsigned i=0;i<vvector->size;++i)

    adition+=gsl_vector_get(vvector,i);

  return adition;

}




//this part is not parallel

void updateweightsandbias(netrbm*ptnet){



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



void meanfieldtrainrbmlogistic(unsigned numepochs,gsl_vector *vectorweightsandbias,
			  gsl_matrix *tdata,unsigned ninputs,unsigned nhidden){



#ifdef _OPENMP
  int maxnumthreads = omp_get_max_threads();
#else /* _OPENMP */
  int maxnumthreads = 4;
#endif /* _OPENMP */

  
  unsigned npatches=tdata->size1;

  const unsigned blocksize=batchsize/maxnumthreads;

  unsigned numbatches=npatches/batchsize;

  netrbm  net(ninputs, nhidden, batchsize,
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


      logistic (&net.blockposprobs[thread].matrix);


      gsl_matrix_memcpy ( &net.blockhiddenactivation[thread].matrix ,&net.blockposprobs[thread].matrix);



      //add visble bias

      for(unsigned i=0;i<net.sizesblocks[thread];++i)
	gsl_matrix_set_row(&net.blockimagedata[thread].matrix,i,&net.visbias.vector);


	//multiply weights
        gsl_blas_dgemm(CblasNoTrans, CblasTrans,
		       1.0, &net.blockhiddenactivation[thread].matrix,
		       &(net.pureweights).matrix,
		       1.0, &net.blockimagedata[thread].matrix);

	//activation function is logistic

	logistic(&net.blockimagedata[thread].matrix);


	//add hidden bias
	for(unsigned i=0;i<net.sizesblocks[thread];++i)
	  gsl_matrix_set_row(&net.blocknegprobs[thread].matrix,i,&net.hidbias.vector);


	//multiply weights
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
		       1.0, &net.blockimagedata[thread].matrix, &net.pureweights.matrix,
		       1.0, &net.blocknegprobs[thread].matrix);


	logistic (&net.blocknegprobs[thread].matrix);


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
