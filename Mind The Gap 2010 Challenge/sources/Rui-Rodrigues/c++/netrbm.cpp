
     /* netrbm.cpp - implements methods of class defined in nerbm.h
        Copyright (C) 2010  Rui Rodrigues <rapr@fct.unl.pt>
        This software is released under the terms of the GNU General
        Public License (http://www.gnu.org/copyleft/gpl.html).
     */

#include "netrbm.h"



netrbm::netrbm(unsigned Ninputs,unsigned Nhidden, unsigned Numcases, 
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




netrbm::~netrbm(){


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

