
     /* parallelgradgsl.cpp - parallel implementation (using openmp) of gradgsl.h
        Copyright (C) 2010  Rui Rodrigues <rapr@fct.unl.pt>
        This software is released under the terms of the GNU General
        Public License (http://www.gnu.org/copyleft/gpl.html).
     */


#include "gradgsl.h"
#include <iostream>
//#include <fstream>

#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;


//---------------------------------------parametersfwdgsl----------------------------------------------------------------------

//constructor

parametersfwdgsl::parametersfwdgsl(int Numhiddenlayers, int*Dimensions,int Batchsize){


  //dimensions is an array with numhiddenlayers+2 elements: ninputs,nhid0,nhid1,...,nhid_numhiddenlayers=noutputs


  numhidlayers=Numhiddenlayers;

  batchsize=Batchsize;

  dimensions=Dimensions;

  ninputs=dimensions[0];

  tam=new int[numhidlayers+1];
  
  tamtotal=0;
  
  for(int i=0;i<numhidlayers+1;++i){

    tam[i]=(dimensions[i]+1)*dimensions[i+1];
    tamtotal+=tam[i];
  }

  initw=new int[numhidlayers+1];

  initw[0]=0;

  for(int i=1;i<numhidlayers+1;++i)
    initw[i]=initw[i-1]+tam[i-1];


  //layerdata[0] is input data hidden layers are 1,2,...,numhid output is  layerdata[numhidlayers+1]
  layerdata=new gsl_matrix*[numhidlayers+1];

  for(int i=0;i<numhidlayers+1;++i){
    layerdata[i]=gsl_matrix_calloc(batchsize,dimensions[i]+1);

    //last column of ones
    gsl_vector_view v=gsl_matrix_column (layerdata[i],dimensions[i] );
    gsl_vector_add_constant (&v.vector, 1.0);
  }

  //here we don't have do add an extra column of ones

  reallayerdata=new gsl_matrix_view[numhidlayers+1];

  for(int i=0;i<numhidlayers+1;++i)
    reallayerdata[i]=gsl_matrix_submatrix(layerdata[i], 0, 0,batchsize , dimensions[i]);
    //eveything except the last column 
  
  
  w=new gsl_matrix_view[numhidlayers+1];

  fwd_data=gsl_matrix_alloc(batchsize,dimensions[numhidlayers+1]);

}

//destructor

parametersfwdgsl::~parametersfwdgsl(){


  delete [] tam;

  delete [] initw;

  for(int i=0;i<numhidlayers+1;++i)
    gsl_matrix_free( layerdata[i]);

  delete [] layerdata;

  delete [] reallayerdata;

  delete [] w;

  gsl_matrix_free(fwd_data);

}


//---------------------------------------parametersgradgsl----------------------------------------------------------------------


//constructor

parametersgradgsl::parametersgradgsl(int Numhiddenlayers, int*Dimensions,int Batchsize){

  //dimensions is an array with numhiddenlayers+2 elements: ninputs,nhid0,nhid1,...,nhid_numhiddenlayers=noutputs


  numhidlayers=Numhiddenlayers;

  batchsize=Batchsize;

  dimensions=Dimensions;

  ninputs=dimensions[0];

  tam=new int[numhidlayers+1];
  
  tamtotal=0;
  
  for(int i=0;i<numhidlayers+1;++i){

    tam[i]=(dimensions[i]+1)*dimensions[i+1];
    tamtotal+=tam[i];
  }

  initw=new int[numhidlayers+1];

  initw[0]=0;

  for(int i=1;i<numhidlayers+1;++i)
    initw[i]=initw[i-1]+tam[i-1];



  //layerdata[0] is input data hidden layers are 1,2,...,numhid output is  layerdata[numhidlayers+1]
  layerdata=new gsl_matrix*[numhidlayers+1];

  for(int i=1;i<numhidlayers+1;++i){
    layerdata[i]=gsl_matrix_calloc(batchsize,dimensions[i]+1);

    //last column of ones
    gsl_vector_view v=gsl_matrix_column (layerdata[i],dimensions[i] );
    gsl_vector_add_constant (&v.vector, 1.0);
  }


 

  lastlayer=gsl_vector_calloc(batchsize*dimensions[numhidlayers+1]);

  matrixlastlayer=gsl_matrix_view_vector(lastlayer,batchsize,dimensions[numhidlayers+1]);
   

  reallayerdata=new gsl_matrix_view[numhidlayers+1];

  for(int i=1;i<numhidlayers+1;++i)
    reallayerdata[i]=gsl_matrix_submatrix(layerdata[i], 0, 0,batchsize , dimensions[i]);
    //eveything except the last column 


  lastlayerdata_matrix=gsl_matrix_alloc(batchsize,dimensions[numhidlayers+1]);  

  w=new gsl_matrix_view[numhidlayers+1];



  aux1=new gsl_matrix*[numhidlayers];
  aux2=new gsl_matrix*[numhidlayers+1];

  for(int i=0;i<numhidlayers;++i){
    aux1[i]=gsl_matrix_calloc(batchsize,dimensions[i+1]);
    aux2[i]=gsl_matrix_calloc(batchsize,dimensions[i+1]);
  }

  aux2[numhidlayers]=gsl_matrix_calloc(batchsize,dimensions[numhidlayers+1]);

  dw=new gsl_matrix_view[numhidlayers+1];

    
  a=new gsl_matrix_view[numhidlayers+1];

}

//destructor

parametersgradgsl::~parametersgradgsl(){


  delete [] tam;

  delete [] initw;



  for(int i=1;i<numhidlayers+1;++i)
    gsl_matrix_free( layerdata[i]);

  gsl_vector_free(lastlayer);

  delete [] layerdata;

  delete [] reallayerdata;

  delete [] w;

  gsl_matrix_free(lastlayerdata_matrix);

  for(int i=0;i<numhidlayers;++i){
    gsl_matrix_free(aux1[i]);
    gsl_matrix_free(aux2[i]);
   }

  gsl_matrix_free(aux2[numhidlayers]);

  delete [] aux1;
  delete [] aux2;

  delete [] dw;

  delete [] a;

}



//------------------------------------end of parametersgradgsl------------------------------------------------------------------

void logistic (gsl_matrix * m,int nrows,int ncols){

  for(int i=0;i<nrows;++i)
    for(int j=0;j<ncols;++j)
      gsl_matrix_set(m,i,j,1/(1+exp((-1)*gsl_matrix_get(m,i,j))));

}



//this is to be used with a big data matrix containing all batches together so uses a smaller set of parameters

void fwdgsl_vislinear(const gsl_vector * x,void * td,gsl_matrix*imagedata){

  parametersfwdgsl*tad=(parametersfwdgsl*) td;
  parametersfwdgsl & T=*tad;



  int numhidlayers=T.numhidlayers;
  


  //_____________________________________________________________________________________
  //openmp
  //
  #ifdef _OPENMP
  int maxnumthreads = omp_get_max_threads();
  #else /* _OPENMP */
  int maxnumthreads = 1;
   #endif /* _OPENMP */

  
  const int blocksize=T.batchsize/maxnumthreads;

  cout<<"maxnumthreads is "<<maxnumthreads<<endl;




  //_____________________________________________________________________________________

  //initial data

  double *xdata=x->data;


  //fill w[hid]

  for(int hid=0;hid<numhidlayers+1;++hid){
  
    T.w[hid] = gsl_matrix_view_array(xdata+T.initw[hid], T.dimensions[hid]+1, T.dimensions[hid+1]);

  }


  

  gsl_matrix_view * blocklayerdata=new gsl_matrix_view[maxnumthreads*(numhidlayers+2)];
  gsl_matrix_view *blockreallayerdata=new gsl_matrix_view[maxnumthreads*(numhidlayers+1)];


  //sizesblocks

  int *sizesblocks=new int[maxnumthreads];

  for(int i=0;i<maxnumthreads-1;++i)
    sizesblocks[i]=blocksize;

  sizesblocks[maxnumthreads-1]=T.batchsize-(maxnumthreads-1)*blocksize;

  //---------



  for(int i=0;i<maxnumthreads;++i){

    for(int hid=0;hid<numhidlayers+1;++hid){

      blocklayerdata[i*(numhidlayers+2)+hid]=gsl_matrix_submatrix(T.layerdata[hid],blocksize*i,0,sizesblocks[i],T.dimensions[hid]+1);

      blockreallayerdata[i*(numhidlayers+1)+hid]=gsl_matrix_submatrix(T.layerdata[hid],blocksize*i,0,sizesblocks[i],T.dimensions[hid]);

    }

    blocklayerdata[i*(numhidlayers+2)+numhidlayers+1]=gsl_matrix_submatrix (imagedata,blocksize*i,0,
									    sizesblocks[i],T.dimensions[numhidlayers+1]);      
  }





#pragma omp parallel for shared(blocklayerdata,blockreallayerdata)
  for(int thread=0;thread<maxnumthreads;++thread){

    for (int hid=0;hid<numhidlayers;++hid){

   
  
      //datahid[i-1]*wi

      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
		     1.0, &blocklayerdata[thread*(numhidlayers+2)+hid].matrix, &T.w[hid].matrix,
		     0.0, &blockreallayerdata[thread*(numhidlayers+1)+hid+1].matrix);

  
      //apply logistic function to the elements of reallayerdata[hid+1]

      logistic (&blockreallayerdata[thread*(numhidlayers+1)+hid+1].matrix,sizesblocks[thread],T.dimensions[hid+1]);

    }


    //last layer activation funstion is identity

    //layerdata[numhidlayers]*w[numhidlayers]

    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
		   1.0, &blocklayerdata[thread*(numhidlayers+2)+numhidlayers].matrix, &T.w[numhidlayers].matrix,
		   0.0, &blocklayerdata[thread*(numhidlayers+2)+numhidlayers+1].matrix);
  }

  delete [] blocklayerdata;

  delete [] blockreallayerdata;

  delete [] sizesblocks;
}



//computing error using parametersfwdgsl (to compute initial and final error)


double just_compute_error_gsl_vislinear(const gsl_vector * x,void * td,gsl_matrix*batchoutputdata) {

  parametersfwdgsl*tad=(parametersfwdgsl*) td;
  parametersfwdgsl & T=*tad;



  int numhidlayers=T.numhidlayers;
  



  //initial data

  double *xdata=x->data;


  //fill w[hid]

  for(int hid=0;hid<numhidlayers+1;++hid){
  
    T.w[hid] = gsl_matrix_view_array(xdata+T.initw[hid], T.dimensions[hid]+1, T.dimensions[hid+1]);

  }



  #ifdef _OPENMP
  int maxnumthreads = omp_get_max_threads();
  #else /* _OPENMP */
  int maxnumthreads = 1;
  #endif /* _OPENMP */

  const int blocksize=T.batchsize/maxnumthreads;


  //sizesblocks----

  int *sizesblocks=new int[maxnumthreads];

  for(int i=0;i<maxnumthreads-1;++i)
    sizesblocks[i]=blocksize;

  sizesblocks[maxnumthreads-1]=T.batchsize-(maxnumthreads-1)*blocksize;

  //-------
    

  gsl_matrix_view * blocklayerdata=new gsl_matrix_view[maxnumthreads*(numhidlayers+2)];
  gsl_matrix_view *blockreallayerdata=new gsl_matrix_view[maxnumthreads*(numhidlayers+1)];
  gsl_matrix_view *blockoutputdata=new gsl_matrix_view[maxnumthreads]; 


#pragma omp parallel for shared(blocklayerdata,blockreallayerdata)
  for(int i=0;i<maxnumthreads;++i){
    for(int hid=0;hid<numhidlayers+1;++hid){

      blocklayerdata[i*(numhidlayers+2)+hid]=gsl_matrix_submatrix(T.layerdata[hid],blocksize*i,0,sizesblocks[i],T.dimensions[hid]+1);

      blockreallayerdata[i*(numhidlayers+1)+hid]=gsl_matrix_submatrix(T.layerdata[hid],blocksize*i,0,sizesblocks[i],T.dimensions[hid]);

    }

    blocklayerdata[i*(numhidlayers+2)+numhidlayers+1]=gsl_matrix_submatrix (T.fwd_data,blocksize*i,0,
									    sizesblocks[i],T.dimensions[numhidlayers+1]);      

    blockoutputdata[i]=gsl_matrix_submatrix (batchoutputdata,blocksize*i,0,sizesblocks[i],T.dimensions[numhidlayers+1]);   
  }

#pragma omp parallel for shared(blocklayerdata,blockreallayerdata)
  for(int thread=0;thread<maxnumthreads;++thread){

    for (int hid=0;hid<numhidlayers;++hid){

     
      //datahid[i-1]*wi

     gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
		     1.0, &blocklayerdata[thread*(numhidlayers+2)+hid].matrix, &T.w[hid].matrix,
		     0.0, &blockreallayerdata[thread*(numhidlayers+1)+hid+1].matrix);

  
      //apply logistic function to the elements of reallayerdata[hid+1]

      logistic (&blockreallayerdata[thread*(numhidlayers+1)+hid+1].matrix,sizesblocks[thread],T.dimensions[hid+1]);

      //logistic (&T.reallayerdata[hid+1].matrix,T.batchsize,T.dimensions[hid+1]);


    }


    //last layer activation funstion is identity

    //layerdata[numhidlayers]*w[numhidlayers]

    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
		   1.0, &blocklayerdata[thread*(numhidlayers+2)+numhidlayers].matrix, &T.w[numhidlayers].matrix,
		   0.0, &blocklayerdata[thread*(numhidlayers+2)+numhidlayers+1].matrix);


    //compute error

    gsl_matrix_sub (&blocklayerdata[thread*(numhidlayers+2)+numhidlayers+1].matrix,
		    &blockoutputdata[thread].matrix);

    //gsl_matrix_sub (T.fwd_data,batchoutputdata);




  }


  delete [] blocklayerdata;

  delete [] blockreallayerdata;
  delete [] blockoutputdata;

  delete [] sizesblocks;


  gsl_vector_const_view vectorimagedata=gsl_vector_const_view_array (T.fwd_data->data, 

								     T.batchsize*T.dimensions[numhidlayers+1]);
  //euclidean norm
  double error=gsl_blas_dnrm2 (&vectorimagedata.vector);

  return error;

  }

//to make a blacklist of patches where the error excedes a limit


void give_error_for_each_patch_gsl_vislinear(const gsl_vector * x,void * td,gsl_matrix*batchoutputdata,double*errorforpatch) {

  parametersfwdgsl*tad=(parametersfwdgsl*) td;
  parametersfwdgsl & T=*tad;



  int numhidlayers=T.numhidlayers;
  



  //initial data

  double *xdata=x->data;


  //fill w[hid]

  for(int hid=0;hid<numhidlayers+1;++hid){
  
    T.w[hid] = gsl_matrix_view_array(xdata+T.initw[hid], T.dimensions[hid]+1, T.dimensions[hid+1]);

  }



  #ifdef _OPENMP
  int maxnumthreads = omp_get_max_threads();
  #else /* _OPENMP */
  int maxnumthreads = 1;
  #endif /* _OPENMP */

  const int blocksize=T.batchsize/maxnumthreads;


  //sizesblocks----

  int *sizesblocks=new int[maxnumthreads];

  for(int i=0;i<maxnumthreads-1;++i)
    sizesblocks[i]=blocksize;

  sizesblocks[maxnumthreads-1]=T.batchsize-(maxnumthreads-1)*blocksize;

  //-------
    

  gsl_matrix_view * blocklayerdata=new gsl_matrix_view[maxnumthreads*(numhidlayers+2)];
  gsl_matrix_view *blockreallayerdata=new gsl_matrix_view[maxnumthreads*(numhidlayers+1)];
  gsl_matrix_view *blockoutputdata=new gsl_matrix_view[maxnumthreads]; 


#pragma omp parallel for shared(blocklayerdata,blockreallayerdata)
  for(int i=0;i<maxnumthreads;++i){
    for(int hid=0;hid<numhidlayers+1;++hid){

      blocklayerdata[i*(numhidlayers+2)+hid]=gsl_matrix_submatrix(T.layerdata[hid],blocksize*i,0,sizesblocks[i],T.dimensions[hid]+1);

      blockreallayerdata[i*(numhidlayers+1)+hid]=gsl_matrix_submatrix(T.layerdata[hid],blocksize*i,0,sizesblocks[i],T.dimensions[hid]);

    }

    blocklayerdata[i*(numhidlayers+2)+numhidlayers+1]=gsl_matrix_submatrix (T.fwd_data,blocksize*i,0,
									    sizesblocks[i],T.dimensions[numhidlayers+1]);      

    blockoutputdata[i]=gsl_matrix_submatrix (batchoutputdata,blocksize*i,0,sizesblocks[i],T.dimensions[numhidlayers+1]);   
  }

#pragma omp parallel for shared(blocklayerdata,blockreallayerdata)
  for(int thread=0;thread<maxnumthreads;++thread){

    for (int hid=0;hid<numhidlayers;++hid){

     
      //datahid[i-1]*wi

     gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
		     1.0, &blocklayerdata[thread*(numhidlayers+2)+hid].matrix, &T.w[hid].matrix,
		     0.0, &blockreallayerdata[thread*(numhidlayers+1)+hid+1].matrix);

  
      //apply logistic function to the elements of reallayerdata[hid+1]

      logistic (&blockreallayerdata[thread*(numhidlayers+1)+hid+1].matrix,sizesblocks[thread],T.dimensions[hid+1]);

      //logistic (&T.reallayerdata[hid+1].matrix,T.batchsize,T.dimensions[hid+1]);


    }


    //last layer activation funstion is identity

    //layerdata[numhidlayers]*w[numhidlayers]

    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
		   1.0, &blocklayerdata[thread*(numhidlayers+2)+numhidlayers].matrix, &T.w[numhidlayers].matrix,
		   0.0, &blocklayerdata[thread*(numhidlayers+2)+numhidlayers+1].matrix);


    //compute error

    gsl_matrix_sub (&blocklayerdata[thread*(numhidlayers+2)+numhidlayers+1].matrix,
		    &blockoutputdata[thread].matrix);

    //gsl_matrix_sub (T.fwd_data,batchoutputdata);




  }


  delete [] blocklayerdata;

  delete [] blockreallayerdata;
  delete [] blockoutputdata;

  delete [] sizesblocks;



  gsl_vector_view rowview;

  for(int i=0;i<(int) T.fwd_data->size1;++i){

    rowview= gsl_matrix_row (T.fwd_data,i);

    errorforpatch[i]=gsl_blas_dnrm2 (&rowview.vector);

  }


}


//------------------------------------------------------------------------------





double compute_error_gsl_vislinear(const gsl_vector * x,void * td) {


  parametersgradgsl*tad=(parametersgradgsl*) td;
  parametersgradgsl & T=*tad;

  int numhidlayers=T.numhidlayers;
  



  //initial data

  double *xdata=x->data;


  //fill w[hid]

  for(int hid=0;hid<numhidlayers+1;++hid){
  
    T.w[hid] = gsl_matrix_view_array(xdata+T.initw[hid], T.dimensions[hid]+1, T.dimensions[hid+1]);

  }


  //--------------------just matrix views to be able to use openmp-----------------------------------------------------

  #ifdef _OPENMP
  int maxnumthreads = omp_get_max_threads();
  #else /* _OPENMP */
  int maxnumthreads = 1;
  #endif /* _OPENMP */

  const int blocksize=T.batchsize/maxnumthreads;


  //sizesblocks----

  int *sizesblocks=new int[maxnumthreads];

  for(int i=0;i<maxnumthreads-1;++i)
    sizesblocks[i]=blocksize;

  sizesblocks[maxnumthreads-1]=T.batchsize-(maxnumthreads-1)*blocksize;

  //-------
    



  gsl_matrix_view * blocklayerdata=new gsl_matrix_view[maxnumthreads*(numhidlayers+2)];
  gsl_matrix_view * blockreallayerdata=new gsl_matrix_view[maxnumthreads*(numhidlayers+1)];

  gsl_matrix_view * blockoutputdata=new gsl_matrix_view[maxnumthreads]; 



#pragma omp parallel for shared(blocklayerdata,blockreallayerdata)
  for(int i=0;i<maxnumthreads;++i){

    //it must be done separate for hid=0 because layerdata[0] is a gsl_matrix_view (submatriz)

    blocklayerdata[i*(numhidlayers+2)]=gsl_matrix_submatrix(T.layerdata[0],blocksize*i,0,sizesblocks[i],T.dimensions[0]+1);
    //blockreallayerdata[0] is never used (I hope)

    for(int hid=1;hid<numhidlayers+1;++hid){

      blocklayerdata[i*(numhidlayers+2)+hid]=gsl_matrix_submatrix(T.layerdata[hid],blocksize*i,0,sizesblocks[i],T.dimensions[hid]+1);

      blockreallayerdata[i*(numhidlayers+1)+hid]=gsl_matrix_submatrix(T.layerdata[hid],blocksize*i,0,sizesblocks[i],T.dimensions[hid]);

    }


    //also must be done separate for last layer

    blocklayerdata[i*(numhidlayers+2)+numhidlayers+1]=gsl_matrix_submatrix (&T.matrixlastlayer.matrix,blocksize*i,0,
									    sizesblocks[i],T.dimensions[numhidlayers+1]);      

    blockoutputdata[i]=gsl_matrix_submatrix (T.batchoutputdata,blocksize*i,0,sizesblocks[i],T.dimensions[numhidlayers+1]);   
  }



#pragma omp parallel for shared(blocklayerdata,blockreallayerdata)
  for(int thread=0;thread<maxnumthreads;++thread){

    for (int hid=0;hid<numhidlayers;++hid){

   
  
      //datahid[i-1]*wi

      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
		     1.0, &blocklayerdata[thread*(numhidlayers+2)+hid].matrix, &T.w[hid].matrix,
		     0.0, &blockreallayerdata[thread*(numhidlayers+1)+hid+1].matrix);

      //apply logistic function to the elements of reallayerdata[hid+1]

      logistic (&blockreallayerdata[thread*(numhidlayers+1)+hid+1].matrix,sizesblocks[thread],T.dimensions[hid+1]);

    }



    //last layer activation funstion is identity

    //layerdata[numhidlayers]*w[numhidlayers]



    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
		   1.0, &blocklayerdata[thread*(numhidlayers+2)+numhidlayers].matrix, &T.w[numhidlayers].matrix,
		   0.0, &blocklayerdata[thread*(numhidlayers+2)+numhidlayers+1].matrix);

    //compute error

    gsl_matrix_sub (&blocklayerdata[thread*(numhidlayers+2)+numhidlayers+1].matrix,
		    &blockoutputdata[thread].matrix);

  }  

    //euclidean norm
      double error=gsl_blas_dnrm2 (T.lastlayer);

      delete [] blocklayerdata;

      delete [] blockreallayerdata;

      delete [] blockoutputdata;

      delete [] sizesblocks;

      return error;
}


//_______________________________________________________________________________________________________________________________
//---------------------------------------------------------------------------------------------------------------------------------
void  gradientgsl_vislinear(const gsl_vector* x,void*td, gsl_vector *grad){


  parametersgradgsl*tad=(parametersgradgsl*) td;
  parametersgradgsl & T=*tad;

  int numhidlayers=T.numhidlayers;
  

  //initial data

  double *xdata=x->data;


  //fill w[hid], bias[hid]

  for(int hid=0;hid<numhidlayers+1;++hid){
  
    T.w[hid] = gsl_matrix_view_array(xdata+T.initw[hid], T.dimensions[hid]+1, T.dimensions[hid+1]);

    T.a[hid]=gsl_matrix_submatrix (&T.w[hid].matrix, 0, 0, T.dimensions[hid],T.dimensions[hid+1]);
  }


  //dw

  int start=0;
  
  for(int i=0;i<numhidlayers+1;++i){

    T.dw[i]=gsl_matrix_view_array (grad->data+start, T.dimensions[i]+1, T.dimensions[i+1]);

    start+=(T.dimensions[i]+1)*T.dimensions[i+1];

  }



  //--------------------just matrix views to be able to use openmp-----------------------------------------------------

  #ifdef _OPENMP
  int maxnumthreads = omp_get_max_threads();
  #else /* _OPENMP */
  int maxnumthreads = 1;
  #endif /* _OPENMP */

  const int blocksize=T.batchsize/maxnumthreads;


  //sizesblocks----

  int *sizesblocks=new int[maxnumthreads];

  for(int i=0;i<maxnumthreads-1;++i)
    sizesblocks[i]=blocksize;

  sizesblocks[maxnumthreads-1]=T.batchsize-(maxnumthreads-1)*blocksize;

  //-------
 


  gsl_matrix_view *blocklayerdata=new gsl_matrix_view[maxnumthreads*(numhidlayers+2)];
  gsl_matrix_view *blockreallayerdata=new gsl_matrix_view[maxnumthreads*(numhidlayers+1)];

  gsl_matrix_view *blockoutputdata=new gsl_matrix_view[maxnumthreads]; 

  gsl_matrix_view * blockaux1=new gsl_matrix_view[maxnumthreads*numhidlayers];
  gsl_matrix_view * blockaux2=new gsl_matrix_view[maxnumthreads*(numhidlayers+1)];



#pragma omp parallel for shared(blocklayerdata,blockreallayerdata)
  for(int thread=0;thread<maxnumthreads;++thread){

    //it must be done separate for hid=0 because layerdata[0] is a gsl_matrix_view (submatriz)

    blocklayerdata[thread*(numhidlayers+2)]=gsl_matrix_submatrix(T.layerdata[0],blocksize*thread,0,sizesblocks[thread],T.dimensions[0]+1);

    blockreallayerdata[thread*(numhidlayers+1)]=gsl_matrix_submatrix(T.layerdata[0],blocksize*thread,0,sizesblocks[thread],T.dimensions[0]);

    for(int hid=1;hid<numhidlayers+1;++hid){

      blocklayerdata[thread*(numhidlayers+2)+hid]=gsl_matrix_submatrix(T.layerdata[hid],blocksize*thread,0,sizesblocks[thread],T.dimensions[hid]+1);

      blockreallayerdata[thread*(numhidlayers+1)+hid]=gsl_matrix_submatrix(T.layerdata[hid],blocksize*thread,0,sizesblocks[thread],T.dimensions[hid]);



    }


    //also must be done separate for last layer

    blocklayerdata[thread*(numhidlayers+2)+numhidlayers+1]=gsl_matrix_submatrix (&T.matrixlastlayer.matrix,blocksize*thread,0,
									    sizesblocks[thread],T.dimensions[numhidlayers+1]);      

    blockoutputdata[thread]=gsl_matrix_submatrix (T.batchoutputdata,blocksize*thread,0,sizesblocks[thread],T.dimensions[numhidlayers+1]);  


    //dw, aux1, aux2---------------
 
    for(int hid=0;hid<numhidlayers;++hid){

      //blockdw[i*(numhidlayers+1)+hid]=gsl_matrix_submatrix(&T.dw[hid].matrix,blocksize*i,0,blocksize,T.dimensions[hid+1]);

      blockaux1[thread*numhidlayers+hid]= gsl_matrix_submatrix(T.aux1[hid],blocksize*thread,0,sizesblocks[thread], T.dimensions[hid+1]);
 
      blockaux2[thread*(numhidlayers+1)+hid]= gsl_matrix_submatrix(T.aux2[hid],blocksize*thread,0,sizesblocks[thread],T.dimensions[hid+1]);

    }


    blockaux2[thread*(numhidlayers+1)+numhidlayers]= gsl_matrix_submatrix(T.aux2[numhidlayers],blocksize*thread,0,
								     sizesblocks[thread],T.dimensions[numhidlayers+1]);
    
  }



#pragma omp parallel for shared(blocklayerdata,blockreallayerdata,blockaux1,blockaux2)
  for(int thread=0;thread<maxnumthreads;++thread){
    for (int hid=0;hid<numhidlayers;++hid){

     
      //datahid[i-1]*wi

      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
		     1.0, &blocklayerdata[thread*(numhidlayers+2)+hid].matrix, &T.w[hid].matrix,
		     0.0, &blockreallayerdata[thread*(numhidlayers+1)+hid+1].matrix);

  
      //apply logistic function to the elements of reallayerdata[hid+1]

      logistic (&blockreallayerdata[thread*(numhidlayers+1)+hid+1].matrix,sizesblocks[thread],T.dimensions[hid+1]);

    }


    //last layer activation funstion is identity

    //layerdata[numhidlayers]*w[numhidlayers]

    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
		   1.0, &blocklayerdata[thread*(numhidlayers+2)+numhidlayers].matrix, &T.w[numhidlayers].matrix,
		   0.0, &blocklayerdata[thread*(numhidlayers+2)+numhidlayers+1].matrix);

    
    //compute error

    gsl_matrix_sub (&blocklayerdata[thread*(numhidlayers+2)+numhidlayers+1].matrix,
		    &blockoutputdata[thread].matrix);


    //to be used in the next layer 
    
    gsl_matrix_memcpy(&blockaux2[thread*(numhidlayers+1)+numhidlayers].matrix,
		      &blocklayerdata[thread*(numhidlayers+2)+numhidlayers+1].matrix);


    for(int hid=numhidlayers-1;hid>-1;--hid){

      gsl_matrix_memcpy (&blockaux1[thread*numhidlayers+hid].matrix,
			 &blockreallayerdata[thread*(numhidlayers+1)+hid+1].matrix);   
      

      gsl_matrix_scale (&blockaux1[thread*numhidlayers+hid].matrix, -1.0);


      gsl_matrix_add_constant (&blockaux1[thread*numhidlayers+hid].matrix, 1.0);


      gsl_matrix_mul_elements (&blockaux1[thread*numhidlayers+hid].matrix,
			       &blockreallayerdata[thread*(numhidlayers+1)+hid+1].matrix);


      gsl_blas_dgemm(CblasNoTrans, CblasTrans,
		     1.0, &blockaux2[thread*(numhidlayers+1)+hid+1].matrix, &T.a[hid+1].matrix,
		     0.0, &blockaux2[thread*(numhidlayers+1)+hid].matrix);
   

      gsl_matrix_mul_elements (&blockaux2[thread*(numhidlayers+1)+hid].matrix,&blockaux1[thread*numhidlayers+hid].matrix);
   

    }

  }

  //update gradients outside the big cycle



  gsl_blas_dgemm(CblasTrans, CblasNoTrans,
		 1.0, T.layerdata[numhidlayers], &T.matrixlastlayer.matrix,
		 0.0, &T.dw[numhidlayers].matrix);


  for(int hid=numhidlayers-1;hid>-1;--hid)

    gsl_blas_dgemm(CblasTrans, CblasNoTrans,
		   1.0, T.layerdata[hid], T.aux2[hid],
		   0.0, &T.dw[hid].matrix);


									  

  //delete [] blockdw;

  delete [] blockaux1;

  delete [] blockaux2;

  delete [] blocklayerdata;

  delete [] blockreallayerdata;

  delete [] blockoutputdata;

  delete [] sizesblocks;

}								  




//---------------------------------------------------------------------------------------------------------------------------------
void  errorandgrad_vislinear(const gsl_vector* x,void*td, double * error, gsl_vector *grad){


  parametersgradgsl*tad=(parametersgradgsl*) td;
  parametersgradgsl & T=*tad;

  int numhidlayers=T.numhidlayers;
  

  //initial data

  double *xdata=x->data;


  //fill w[hid], bias[hid]

  for(int hid=0;hid<numhidlayers+1;++hid){
  
    T.w[hid] = gsl_matrix_view_array(xdata+T.initw[hid], T.dimensions[hid]+1, T.dimensions[hid+1]);

    T.a[hid]=gsl_matrix_submatrix (&T.w[hid].matrix, 0, 0, T.dimensions[hid],T.dimensions[hid+1]);
  }


 //dw

  int start=0;
  
  for(int i=0;i<numhidlayers+1;++i){

    T.dw[i]=gsl_matrix_view_array (grad->data+start, T.dimensions[i]+1, T.dimensions[i+1]);

    start+=(T.dimensions[i]+1)*T.dimensions[i+1];

  }



  //--------------------just matrix views to be able to use openmp-----------------------------------------------------

  #ifdef _OPENMP
  int maxnumthreads = omp_get_max_threads();
  #else /* _OPENMP */
  int maxnumthreads = 1;
  #endif /* _OPENMP */

  const int blocksize=T.batchsize/maxnumthreads;


  //sizesblocks----

  int *sizesblocks=new int[maxnumthreads];

  for(int i=0;i<maxnumthreads-1;++i)
    sizesblocks[i]=blocksize;

  sizesblocks[maxnumthreads-1]=T.batchsize-(maxnumthreads-1)*blocksize;

  //-------



  gsl_matrix_view *blocklayerdata=new gsl_matrix_view[maxnumthreads*(numhidlayers+2)];
  gsl_matrix_view *blockreallayerdata=new gsl_matrix_view[maxnumthreads*(numhidlayers+1)];

  gsl_matrix_view *blockoutputdata=new gsl_matrix_view[maxnumthreads]; 

  gsl_matrix_view * blockaux1=new gsl_matrix_view[maxnumthreads*numhidlayers];
  gsl_matrix_view * blockaux2=new gsl_matrix_view[maxnumthreads*(numhidlayers+1)];



#pragma omp parallel for shared(blocklayerdata,blockreallayerdata)
  for(int thread=0;thread<maxnumthreads;++thread){

    //it must be done separate for hid=0 because layerdata[0] is a gsl_matrix_view (submatriz)

    blocklayerdata[thread*(numhidlayers+2)]=gsl_matrix_submatrix(T.layerdata[0],blocksize*thread,0,sizesblocks[thread],T.dimensions[0]+1);

    blockreallayerdata[thread*(numhidlayers+1)]=gsl_matrix_submatrix(T.layerdata[0],blocksize*thread,0,sizesblocks[thread],T.dimensions[0]);

    for(int hid=1;hid<numhidlayers+1;++hid){

      blocklayerdata[thread*(numhidlayers+2)+hid]=gsl_matrix_submatrix(T.layerdata[hid],blocksize*thread,0,sizesblocks[thread],T.dimensions[hid]+1);

      blockreallayerdata[thread*(numhidlayers+1)+hid]=gsl_matrix_submatrix(T.layerdata[hid],blocksize*thread,0,sizesblocks[thread],T.dimensions[hid]);



    }


    //also must be done separate for last layer

    blocklayerdata[thread*(numhidlayers+2)+numhidlayers+1]=gsl_matrix_submatrix (&T.matrixlastlayer.matrix,blocksize*thread,0,
									    sizesblocks[thread],T.dimensions[numhidlayers+1]);      

    blockoutputdata[thread]=gsl_matrix_submatrix (T.batchoutputdata,blocksize*thread,0,sizesblocks[thread],T.dimensions[numhidlayers+1]);  


    //dw, aux1, aux2---------------
 
    for(int hid=0;hid<numhidlayers;++hid){


      blockaux1[thread*numhidlayers+hid]= gsl_matrix_submatrix(T.aux1[hid],blocksize*thread,0,sizesblocks[thread], T.dimensions[hid+1]);
 
      blockaux2[thread*(numhidlayers+1)+hid]= gsl_matrix_submatrix(T.aux2[hid],blocksize*thread,0,sizesblocks[thread],T.dimensions[hid+1]);

    }

    blockaux2[thread*(numhidlayers+1)+numhidlayers]= gsl_matrix_submatrix(T.aux2[numhidlayers],blocksize*thread,0,
								     sizesblocks[thread],T.dimensions[numhidlayers+1]);    
  }


  //-----------------------------------------------------------------------------------------------





#pragma omp parallel for shared(blocklayerdata,blockreallayerdata,blockaux1,blockaux2)
  for(int thread=0;thread<maxnumthreads;++thread){
    for (int hid=0;hid<numhidlayers;++hid){

     
      //datahid[i-1]*wi

      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
		     1.0, &blocklayerdata[thread*(numhidlayers+2)+hid].matrix, &T.w[hid].matrix,
		     0.0, &blockreallayerdata[thread*(numhidlayers+1)+hid+1].matrix);

  
      //apply logistic function to the elements of reallayerdata[hid+1]

      logistic (&blockreallayerdata[thread*(numhidlayers+1)+hid+1].matrix,sizesblocks[thread],T.dimensions[hid+1]);
    }


    //last layer activation funstion is identity

    //layerdata[numhidlayers]*w[numhidlayers]

    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
		   1.0, &blocklayerdata[thread*(numhidlayers+2)+numhidlayers].matrix, &T.w[numhidlayers].matrix,
		   0.0, &blocklayerdata[thread*(numhidlayers+2)+numhidlayers+1].matrix);

    
    //compute error

    gsl_matrix_sub (&blocklayerdata[thread*(numhidlayers+2)+numhidlayers+1].matrix,
		    &blockoutputdata[thread].matrix);


    
    gsl_matrix_memcpy(&blockaux2[thread*(numhidlayers+1)+numhidlayers].matrix,
		      &blocklayerdata[thread*(numhidlayers+2)+numhidlayers+1].matrix);

    //T.aux2[numhidlayers]=&T.matrixlastlayer.matrix;



    for(int hid=numhidlayers-1;hid>-1;--hid){

      gsl_matrix_memcpy (&blockaux1[thread*numhidlayers+hid].matrix,
			 &blockreallayerdata[thread*(numhidlayers+1)+hid+1].matrix);   
      

      gsl_matrix_scale (&blockaux1[thread*numhidlayers+hid].matrix, -1.0);


      gsl_matrix_add_constant (&blockaux1[thread*numhidlayers+hid].matrix, 1.0);


      gsl_matrix_mul_elements (&blockaux1[thread*numhidlayers+hid].matrix,
			       &blockreallayerdata[thread*(numhidlayers+1)+hid+1].matrix);


      gsl_blas_dgemm(CblasNoTrans, CblasTrans,
		     1.0, &blockaux2[thread*(numhidlayers+1)+hid+1].matrix, &T.a[hid+1].matrix,
		     0.0, &blockaux2[thread*(numhidlayers+1)+hid].matrix);
   

      gsl_matrix_mul_elements (&blockaux2[thread*(numhidlayers+1)+hid].matrix,&blockaux1[thread*numhidlayers+hid].matrix);
   

    }

  }

  //update gradients outside the big cycle



  gsl_blas_dgemm(CblasTrans, CblasNoTrans,
		 1.0, T.layerdata[numhidlayers], &T.matrixlastlayer.matrix,
		 0.0, &T.dw[numhidlayers].matrix);


  for(int hid=numhidlayers-1;hid>-1;--hid)

    gsl_blas_dgemm(CblasTrans, CblasNoTrans,
		   1.0, T.layerdata[hid], T.aux2[hid],
		   0.0, &T.dw[hid].matrix);


  //euclidean norm
  *error=gsl_blas_dnrm2 (T.lastlayer);


  delete [] blockaux1;

  delete [] blockaux2;

  delete [] blocklayerdata;

  delete [] blockreallayerdata;

  delete [] blockoutputdata;
  
  delete [] sizesblocks;

}



//------------------------------------------------------------------------------------------------
//when the activation of last layer is logistic (and not the identity) ---------------------------
//________________________________________________________________________________________________


void fwdgsl_vislogistic(const gsl_vector * x,void * td,gsl_matrix*imagedata){



  parametersfwdgsl*tad=(parametersfwdgsl*) td;
  parametersfwdgsl & T=*tad;



  int numhidlayers=T.numhidlayers;
  


  //_____________________________________________________________________________________
  //openmp
  //
  #ifdef _OPENMP
  int maxnumthreads = omp_get_max_threads();
  #else /* _OPENMP */
  int maxnumthreads = 1;
   #endif /* _OPENMP */

  
  const int blocksize=T.batchsize/maxnumthreads;

  cout<<"maxnumthreads is "<<maxnumthreads<<endl;




  //_____________________________________________________________________________________

  //initial data

  double *xdata=x->data;


  //fill w[hid]

  for(int hid=0;hid<numhidlayers+1;++hid){
  
    T.w[hid] = gsl_matrix_view_array(xdata+T.initw[hid], T.dimensions[hid]+1, T.dimensions[hid+1]);

  }


  

  gsl_matrix_view * blocklayerdata=new gsl_matrix_view[maxnumthreads*(numhidlayers+2)];
  gsl_matrix_view *blockreallayerdata=new gsl_matrix_view[maxnumthreads*(numhidlayers+1)];


  //sizesblocks

  int *sizesblocks=new int[maxnumthreads];

  for(int i=0;i<maxnumthreads-1;++i)
    sizesblocks[i]=blocksize;

  sizesblocks[maxnumthreads-1]=T.batchsize-(maxnumthreads-1)*blocksize;

  //---------



  for(int i=0;i<maxnumthreads;++i){

    for(int hid=0;hid<numhidlayers+1;++hid){

      blocklayerdata[i*(numhidlayers+2)+hid]=gsl_matrix_submatrix(T.layerdata[hid],blocksize*i,0,sizesblocks[i],T.dimensions[hid]+1);

      blockreallayerdata[i*(numhidlayers+1)+hid]=gsl_matrix_submatrix(T.layerdata[hid],blocksize*i,0,sizesblocks[i],T.dimensions[hid]);

    }

    blocklayerdata[i*(numhidlayers+2)+numhidlayers+1]=gsl_matrix_submatrix (imagedata,blocksize*i,0,
									    sizesblocks[i],T.dimensions[numhidlayers+1]);      
  }





#pragma omp parallel for shared(blocklayerdata,blockreallayerdata)
  for(int thread=0;thread<maxnumthreads;++thread){

    for (int hid=0;hid<numhidlayers;++hid){

   
  
      //datahid[i-1]*wi

      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
		     1.0, &blocklayerdata[thread*(numhidlayers+2)+hid].matrix, &T.w[hid].matrix,
		     0.0, &blockreallayerdata[thread*(numhidlayers+1)+hid+1].matrix);

  
      //apply logistic function to the elements of reallayerdata[hid+1]

      logistic (&blockreallayerdata[thread*(numhidlayers+1)+hid+1].matrix,sizesblocks[thread],T.dimensions[hid+1]);

    }


    //layerdata[numhidlayers]*w[numhidlayers]

    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
    	   1.0, &blocklayerdata[thread*(numhidlayers+2)+numhidlayers].matrix, &T.w[numhidlayers].matrix,
    	   0.0, &blocklayerdata[thread*(numhidlayers+2)+numhidlayers+1].matrix);


    //activation function for last layer is logistic


    logistic (&blocklayerdata[thread*(numhidlayers+2)+numhidlayers+1].matrix,sizesblocks[thread],T.dimensions[numhidlayers+1]);

  }

  delete [] blocklayerdata;

  delete [] blockreallayerdata;

  delete [] sizesblocks;

 
}




//computing error using parametersfwdgsl (to compute initial and final error)


double just_compute_error_gsl_vislogistic(const gsl_vector * x,
					  void * td,gsl_matrix*batchoutputdata) {



  parametersfwdgsl*tad=(parametersfwdgsl*) td;
  parametersfwdgsl & T=*tad;



  int numhidlayers=T.numhidlayers;
  



  //initial data

  double *xdata=x->data;


  //fill w[hid]

  for(int hid=0;hid<numhidlayers+1;++hid){
  
    T.w[hid] = gsl_matrix_view_array(xdata+T.initw[hid], T.dimensions[hid]+1, T.dimensions[hid+1]);

  }



  #ifdef _OPENMP
  int maxnumthreads = omp_get_max_threads();
  #else /* _OPENMP */
  int maxnumthreads = 1;
  #endif /* _OPENMP */

  const int blocksize=T.batchsize/maxnumthreads;


  //sizesblocks----

  int *sizesblocks=new int[maxnumthreads];

  for(int i=0;i<maxnumthreads-1;++i)
    sizesblocks[i]=blocksize;

  sizesblocks[maxnumthreads-1]=T.batchsize-(maxnumthreads-1)*blocksize;

  //-------
    

  gsl_matrix_view * blocklayerdata=new gsl_matrix_view[maxnumthreads*(numhidlayers+2)];
  gsl_matrix_view *blockreallayerdata=new gsl_matrix_view[maxnumthreads*(numhidlayers+1)];
  gsl_matrix_view *blockoutputdata=new gsl_matrix_view[maxnumthreads]; 


#pragma omp parallel for shared(blocklayerdata,blockreallayerdata)
  for(int i=0;i<maxnumthreads;++i){
    for(int hid=0;hid<numhidlayers+1;++hid){

      blocklayerdata[i*(numhidlayers+2)+hid]=gsl_matrix_submatrix(T.layerdata[hid],blocksize*i,0,sizesblocks[i],T.dimensions[hid]+1);

      blockreallayerdata[i*(numhidlayers+1)+hid]=gsl_matrix_submatrix(T.layerdata[hid],blocksize*i,0,sizesblocks[i],T.dimensions[hid]);

    }

    blocklayerdata[i*(numhidlayers+2)+numhidlayers+1]=gsl_matrix_submatrix (T.fwd_data,blocksize*i,0,
									    sizesblocks[i],T.dimensions[numhidlayers+1]);      

    blockoutputdata[i]=gsl_matrix_submatrix (batchoutputdata,blocksize*i,0,sizesblocks[i],T.dimensions[numhidlayers+1]);   
  }

#pragma omp parallel for shared(blocklayerdata,blockreallayerdata)
  for(int thread=0;thread<maxnumthreads;++thread){

    for (int hid=0;hid<numhidlayers;++hid){

     
      //datahid[i-1]*wi

     gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
		     1.0, &blocklayerdata[thread*(numhidlayers+2)+hid].matrix, &T.w[hid].matrix,
		     0.0, &blockreallayerdata[thread*(numhidlayers+1)+hid+1].matrix);

  
      //apply logistic function to the elements of reallayerdata[hid+1]

      logistic (&blockreallayerdata[thread*(numhidlayers+1)+hid+1].matrix,sizesblocks[thread],T.dimensions[hid+1]);


    }

    //last layer
    //layerdata[numhidlayers]*w[numhidlayers]
    
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
    	   1.0, &blocklayerdata[thread*(numhidlayers+2)+numhidlayers].matrix, &T.w[numhidlayers].matrix,
    	   0.0, &blocklayerdata[thread*(numhidlayers+2)+numhidlayers+1].matrix);

    logistic (&blocklayerdata[thread*(numhidlayers+2)+numhidlayers+1].matrix,sizesblocks[thread],T.dimensions[numhidlayers+1]);


    //compute error

    gsl_matrix_sub (&blocklayerdata[thread*(numhidlayers+2)+numhidlayers+1].matrix,
		    &blockoutputdata[thread].matrix);

  }


  delete [] blocklayerdata;

  delete [] blockreallayerdata;
  delete [] blockoutputdata;

  delete [] sizesblocks;


  gsl_vector_const_view vectorimagedata=gsl_vector_const_view_array (T.fwd_data->data, 

								     T.batchsize*T.dimensions[numhidlayers+1]);

  //euclidean norm
  double error=gsl_blas_dnrm2 (&vectorimagedata.vector);

  return error;


}







double compute_error_gsl_vislogistic(const gsl_vector * x,void * td) {



  parametersgradgsl*tad=(parametersgradgsl*) td;
  parametersgradgsl & T=*tad;

  int numhidlayers=T.numhidlayers;
  



  //initial data

  double *xdata=x->data;


  //fill w[hid]

  for(int hid=0;hid<numhidlayers+1;++hid){
  
    T.w[hid] = gsl_matrix_view_array(xdata+T.initw[hid], T.dimensions[hid]+1, T.dimensions[hid+1]);

  }


  //--------------------just matrix views to be able to use openmp-----------------------------------------------------

  #ifdef _OPENMP
  int maxnumthreads = omp_get_max_threads();
  #else /* _OPENMP */
  int maxnumthreads = 1;
  #endif /* _OPENMP */

  const int blocksize=T.batchsize/maxnumthreads;


  //sizesblocks----

  int *sizesblocks=new int[maxnumthreads];

  for(int i=0;i<maxnumthreads-1;++i)
    sizesblocks[i]=blocksize;

  sizesblocks[maxnumthreads-1]=T.batchsize-(maxnumthreads-1)*blocksize;

  //-------
    



  gsl_matrix_view * blocklayerdata=new gsl_matrix_view[maxnumthreads*(numhidlayers+2)];
  gsl_matrix_view * blockreallayerdata=new gsl_matrix_view[maxnumthreads*(numhidlayers+1)];

  gsl_matrix_view * blockoutputdata=new gsl_matrix_view[maxnumthreads]; 



#pragma omp parallel for shared(blocklayerdata,blockreallayerdata)
  for(int i=0;i<maxnumthreads;++i){

    //it must be done separate for hid=0 because layerdata[0] is a gsl_matrix_view (submatriz)

    blocklayerdata[i*(numhidlayers+2)]=gsl_matrix_submatrix(T.layerdata[0],blocksize*i,0,sizesblocks[i],T.dimensions[0]+1);
    //blockreallayerdata[0] is never used (I hope)

    for(int hid=1;hid<numhidlayers+1;++hid){

      blocklayerdata[i*(numhidlayers+2)+hid]=gsl_matrix_submatrix(T.layerdata[hid],blocksize*i,0,sizesblocks[i],T.dimensions[hid]+1);

      blockreallayerdata[i*(numhidlayers+1)+hid]=gsl_matrix_submatrix(T.layerdata[hid],blocksize*i,0,sizesblocks[i],T.dimensions[hid]);

    }


    //also must be done separate for last layer

    blocklayerdata[i*(numhidlayers+2)+numhidlayers+1]=gsl_matrix_submatrix (&T.matrixlastlayer.matrix,blocksize*i,0,
									    sizesblocks[i],T.dimensions[numhidlayers+1]);      

    blockoutputdata[i]=gsl_matrix_submatrix (T.batchoutputdata,blocksize*i,0,sizesblocks[i],T.dimensions[numhidlayers+1]);   
  }



#pragma omp parallel for shared(blocklayerdata,blockreallayerdata)
  for(int thread=0;thread<maxnumthreads;++thread){

    for (int hid=0;hid<numhidlayers;++hid){

   
  
      //datahid[i-1]*wi

      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
		     1.0, &blocklayerdata[thread*(numhidlayers+2)+hid].matrix, &T.w[hid].matrix,
		     0.0, &blockreallayerdata[thread*(numhidlayers+1)+hid+1].matrix);


  
      //apply logistic function to the elements of reallayerdata[hid+1]

      logistic (&blockreallayerdata[thread*(numhidlayers+1)+hid+1].matrix,sizesblocks[thread],T.dimensions[hid+1]);

    }



    //last layer activation funstion is not identity

    //layerdata[numhidlayers]*w[numhidlayers]



    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
    	   1.0, &blocklayerdata[thread*(numhidlayers+2)+numhidlayers].matrix, &T.w[numhidlayers].matrix,
    	   0.0, &blocklayerdata[thread*(numhidlayers+2)+numhidlayers+1].matrix);


    logistic (&blocklayerdata[thread*(numhidlayers+2)+numhidlayers+1].matrix,
	      sizesblocks[thread],T.dimensions[numhidlayers+1]);



    //compute error

    gsl_matrix_sub (&blocklayerdata[thread*(numhidlayers+2)+numhidlayers+1].matrix,
		    &blockoutputdata[thread].matrix);
  }  
    //euclidean norm
      double error=gsl_blas_dnrm2 (T.lastlayer);

      delete [] blocklayerdata;

      delete [] blockreallayerdata;

      delete [] blockoutputdata;

      delete [] sizesblocks;

      return error;
  
}



//---------------------------------------------------------------------------------------------------------



void  gradientgsl_vislogistic(const gsl_vector* x,void*td, gsl_vector *grad){



  parametersgradgsl*tad=(parametersgradgsl*) td;
  parametersgradgsl & T=*tad;

  int numhidlayers=T.numhidlayers;
  

  //initial data

  double *xdata=x->data;


  //fill w[hid], bias[hid]

  for(int hid=0;hid<numhidlayers+1;++hid){
  
    T.w[hid] = gsl_matrix_view_array(xdata+T.initw[hid], T.dimensions[hid]+1, T.dimensions[hid+1]);

    T.a[hid]=gsl_matrix_submatrix (&T.w[hid].matrix, 0, 0, T.dimensions[hid],T.dimensions[hid+1]);
  }


  //dw

  int start=0;
  
  for(int i=0;i<numhidlayers+1;++i){

    T.dw[i]=gsl_matrix_view_array (grad->data+start, T.dimensions[i]+1, T.dimensions[i+1]);

    start+=(T.dimensions[i]+1)*T.dimensions[i+1];

  }



  //--------------------just matrix views to be able to use openmp-----------------------------------------------------

  #ifdef _OPENMP
  int maxnumthreads = omp_get_max_threads();
  #else /* _OPENMP */
  int maxnumthreads = 1;
  #endif /* _OPENMP */

  const int blocksize=T.batchsize/maxnumthreads;


  //sizesblocks----

  int *sizesblocks=new int[maxnumthreads];

  for(int i=0;i<maxnumthreads-1;++i)
    sizesblocks[i]=blocksize;

  sizesblocks[maxnumthreads-1]=T.batchsize-(maxnumthreads-1)*blocksize;

  //-------
 


  gsl_matrix_view *blocklayerdata=new gsl_matrix_view[maxnumthreads*(numhidlayers+2)];
  gsl_matrix_view *blockreallayerdata=new gsl_matrix_view[maxnumthreads*(numhidlayers+1)];

  gsl_matrix_view *blockoutputdata=new gsl_matrix_view[maxnumthreads]; 

  gsl_matrix_view * blockaux1=new gsl_matrix_view[maxnumthreads*numhidlayers];
  gsl_matrix_view * blockaux2=new gsl_matrix_view[maxnumthreads*(numhidlayers+1)];

  gsl_matrix_view *blocklastlayerdata_matrix=new gsl_matrix_view[maxnumthreads];

#pragma omp parallel for shared(blocklayerdata,blockreallayerdata)
  for(int thread=0;thread<maxnumthreads;++thread){

    //it must be done separate for hid=0 because layerdata[0] is a gsl_matrix_view (submatriz)

    blocklayerdata[thread*(numhidlayers+2)]=gsl_matrix_submatrix(T.layerdata[0],blocksize*thread,0,
								 sizesblocks[thread],T.dimensions[0]+1);

    blockreallayerdata[thread*(numhidlayers+1)]=gsl_matrix_submatrix(T.layerdata[0],blocksize*thread,0,
								     sizesblocks[thread],T.dimensions[0]);

    for(int hid=1;hid<numhidlayers+1;++hid){

      blocklayerdata[thread*(numhidlayers+2)+hid]=gsl_matrix_submatrix(T.layerdata[hid],blocksize*thread,0,
								       sizesblocks[thread],T.dimensions[hid]+1);

      blockreallayerdata[thread*(numhidlayers+1)+hid]=gsl_matrix_submatrix(T.layerdata[hid],blocksize*thread,0,
									   sizesblocks[thread],T.dimensions[hid]);
    }


    //must be done separate for last layer

    blocklayerdata[thread*(numhidlayers+2)+numhidlayers+1]=gsl_matrix_submatrix (&T.matrixlastlayer.matrix,blocksize*thread,0,
									    sizesblocks[thread],T.dimensions[numhidlayers+1]);      

    blockoutputdata[thread]=gsl_matrix_submatrix (T.batchoutputdata,blocksize*thread,0,
						  sizesblocks[thread],T.dimensions[numhidlayers+1]);  

    blocklastlayerdata_matrix[thread]=gsl_matrix_submatrix (T.lastlayerdata_matrix,blocksize*thread,0,
							    sizesblocks[thread],T.dimensions[numhidlayers+1]);  


    //dw, aux1, aux2---------------
 
    for(int hid=0;hid<numhidlayers;++hid){

      blockaux1[thread*numhidlayers+hid]= gsl_matrix_submatrix(T.aux1[hid],blocksize*thread,0,
							       sizesblocks[thread], T.dimensions[hid+1]);
 
      blockaux2[thread*(numhidlayers+1)+hid]= gsl_matrix_submatrix(T.aux2[hid],blocksize*thread,0,
								   sizesblocks[thread],T.dimensions[hid+1]);

    }


    blockaux2[thread*(numhidlayers+1)+numhidlayers]= gsl_matrix_submatrix(T.aux2[numhidlayers],blocksize*thread,0,
								     sizesblocks[thread],T.dimensions[numhidlayers+1]);
    
  }



#pragma omp parallel for shared(blocklayerdata,blockreallayerdata,blockaux1,blockaux2)
  for(int thread=0;thread<maxnumthreads;++thread){
    for (int hid=0;hid<numhidlayers;++hid){

     
      //datahid[i-1]*wi

      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
		     1.0, &blocklayerdata[thread*(numhidlayers+2)+hid].matrix, &T.w[hid].matrix,
		     0.0, &blockreallayerdata[thread*(numhidlayers+1)+hid+1].matrix);

  
      //apply logistic function to the elements of reallayerdata[hid+1]

      logistic (&blockreallayerdata[thread*(numhidlayers+1)+hid+1].matrix,sizesblocks[thread],T.dimensions[hid+1]);

    }


    //last layer activation funstion is not identity

    //layerdata[numhidlayers]*w[numhidlayers]

     gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
 		   1.0, &blocklayerdata[thread*(numhidlayers+2)+numhidlayers].matrix, &T.w[numhidlayers].matrix,
 		   0.0, &blocklayerdata[thread*(numhidlayers+2)+numhidlayers+1].matrix);

     logistic (&blocklayerdata[thread*(numhidlayers+2)+numhidlayers+1].matrix,sizesblocks[thread],T.dimensions[numhidlayers+1]);


    gsl_matrix_memcpy (&blocklastlayerdata_matrix[thread].matrix,
		       &blocklayerdata[thread*(numhidlayers+2)+numhidlayers+1].matrix);    

    
    //compute error matrix

    gsl_matrix_sub (&blocklayerdata[thread*(numhidlayers+2)+numhidlayers+1].matrix,
		    &blockoutputdata[thread].matrix);

    
    gsl_matrix_memcpy(&blockaux2[thread*(numhidlayers+1)+numhidlayers].matrix,
		      &blocklayerdata[thread*(numhidlayers+2)+numhidlayers+1].matrix);

    gsl_matrix_mul_elements (&blockaux2[thread*(numhidlayers+1)+numhidlayers].matrix
			     ,&blocklastlayerdata_matrix[thread].matrix);


    gsl_matrix_scale (&blocklastlayerdata_matrix[thread].matrix, -1.0);


    gsl_matrix_add_constant(&blocklastlayerdata_matrix[thread].matrix, 1.0);


    gsl_matrix_mul_elements (&blockaux2[thread*(numhidlayers+1)+numhidlayers].matrix,
			     &blocklastlayerdata_matrix[thread].matrix);

    for(int hid=numhidlayers-1;hid>-1;--hid){

      gsl_matrix_memcpy (&blockaux1[thread*numhidlayers+hid].matrix,&blockreallayerdata[thread*(numhidlayers+1)+hid+1].matrix);   
      

      gsl_matrix_scale (&blockaux1[thread*numhidlayers+hid].matrix, -1.0);


      gsl_matrix_add_constant (&blockaux1[thread*numhidlayers+hid].matrix, 1.0);


      gsl_matrix_mul_elements (&blockaux1[thread*numhidlayers+hid].matrix,&blockreallayerdata[thread*(numhidlayers+1)+hid+1].matrix);


      gsl_blas_dgemm(CblasNoTrans, CblasTrans,
		     1.0, &blockaux2[thread*(numhidlayers+1)+hid+1].matrix, &T.a[hid+1].matrix,
		     0.0, &blockaux2[thread*(numhidlayers+1)+hid].matrix);
   

      gsl_matrix_mul_elements (&blockaux2[thread*(numhidlayers+1)+hid].matrix,&blockaux1[thread*numhidlayers+hid].matrix);
   

    }

  }

  //update gradients outside the big cycle


  for(int hid=numhidlayers;hid>-1;--hid)

    gsl_blas_dgemm(CblasTrans, CblasNoTrans,
		   1.0, T.layerdata[hid], T.aux2[hid],
		   0.0, &T.dw[hid].matrix);


									  

  //delete [] blockdw;

  delete [] blockaux1;

  delete [] blockaux2;

  delete [] blocklayerdata;

  delete [] blockreallayerdata;

  delete [] blockoutputdata;

  delete [] sizesblocks;

  delete []   blocklastlayerdata_matrix;

 
}


//----------------------------------------------------------------------------------------------------

void  errorandgrad_vislogistic(const gsl_vector* x,void*td, double * error, gsl_vector *grad){




  parametersgradgsl*tad=(parametersgradgsl*) td;
  parametersgradgsl & T=*tad;

  int numhidlayers=T.numhidlayers;
  

  //initial data

  double *xdata=x->data;


  //fill w[hid], bias[hid]

  for(int hid=0;hid<numhidlayers+1;++hid){
  
    T.w[hid] = gsl_matrix_view_array(xdata+T.initw[hid], T.dimensions[hid]+1, T.dimensions[hid+1]);

    T.a[hid]=gsl_matrix_submatrix (&T.w[hid].matrix, 0, 0, T.dimensions[hid],T.dimensions[hid+1]);
  }


  //dw

  int start=0;
  
  for(int i=0;i<numhidlayers+1;++i){

    T.dw[i]=gsl_matrix_view_array (grad->data+start, T.dimensions[i]+1, T.dimensions[i+1]);

    start+=(T.dimensions[i]+1)*T.dimensions[i+1];

  }



  //--------------------just matrix views to be able to use openmp-----------------------------------------------------

  #ifdef _OPENMP
  int maxnumthreads = omp_get_max_threads();
  #else /* _OPENMP */
  int maxnumthreads = 1;
  #endif /* _OPENMP */

  const int blocksize=T.batchsize/maxnumthreads;


  //sizesblocks----

  int *sizesblocks=new int[maxnumthreads];

  for(int i=0;i<maxnumthreads-1;++i)
    sizesblocks[i]=blocksize;

  sizesblocks[maxnumthreads-1]=T.batchsize-(maxnumthreads-1)*blocksize;

  //-------
 


  gsl_matrix_view *blocklayerdata=new gsl_matrix_view[maxnumthreads*(numhidlayers+2)];
  gsl_matrix_view *blockreallayerdata=new gsl_matrix_view[maxnumthreads*(numhidlayers+1)];

  gsl_matrix_view *blockoutputdata=new gsl_matrix_view[maxnumthreads]; 

  gsl_matrix_view * blockaux1=new gsl_matrix_view[maxnumthreads*numhidlayers];
  gsl_matrix_view * blockaux2=new gsl_matrix_view[maxnumthreads*(numhidlayers+1)];

  gsl_matrix_view *blocklastlayerdata_matrix=new gsl_matrix_view[maxnumthreads];

#pragma omp parallel for shared(blocklayerdata,blockreallayerdata)
  for(int thread=0;thread<maxnumthreads;++thread){

    //it must be done separate for hid=0 because layerdata[0] is a gsl_matrix_view (submatriz)

    blocklayerdata[thread*(numhidlayers+2)]=gsl_matrix_submatrix(T.layerdata[0],blocksize*thread,0,sizesblocks[thread],T.dimensions[0]+1);

    blockreallayerdata[thread*(numhidlayers+1)]=gsl_matrix_submatrix(T.layerdata[0],blocksize*thread,0,sizesblocks[thread],T.dimensions[0]);

    for(int hid=1;hid<numhidlayers+1;++hid){

      blocklayerdata[thread*(numhidlayers+2)+hid]=gsl_matrix_submatrix(T.layerdata[hid],blocksize*thread,0,sizesblocks[thread],T.dimensions[hid]+1);

      blockreallayerdata[thread*(numhidlayers+1)+hid]=gsl_matrix_submatrix(T.layerdata[hid],blocksize*thread,0,sizesblocks[thread],T.dimensions[hid]);



    }


    //must be done separate for last layer

    blocklayerdata[thread*(numhidlayers+2)+numhidlayers+1]=gsl_matrix_submatrix (&T.matrixlastlayer.matrix,blocksize*thread,0,
									    sizesblocks[thread],T.dimensions[numhidlayers+1]);      

    blockoutputdata[thread]=gsl_matrix_submatrix (T.batchoutputdata,blocksize*thread,0,
						  sizesblocks[thread],T.dimensions[numhidlayers+1]);  

    blocklastlayerdata_matrix[thread]=gsl_matrix_submatrix (T.lastlayerdata_matrix,blocksize*thread,0,
							    sizesblocks[thread],T.dimensions[numhidlayers+1]);  


    //dw, aux1, aux2---------------
 
    for(int hid=0;hid<numhidlayers;++hid){

      //blockdw[i*(numhidlayers+1)+hid]=gsl_matrix_submatrix(&T.dw[hid].matrix,blocksize*i,0,blocksize,T.dimensions[hid+1]);

      blockaux1[thread*numhidlayers+hid]= gsl_matrix_submatrix(T.aux1[hid],blocksize*thread,0,sizesblocks[thread], T.dimensions[hid+1]);
 
      blockaux2[thread*(numhidlayers+1)+hid]= gsl_matrix_submatrix(T.aux2[hid],blocksize*thread,0,sizesblocks[thread],T.dimensions[hid+1]);

    }


    blockaux2[thread*(numhidlayers+1)+numhidlayers]= gsl_matrix_submatrix(T.aux2[numhidlayers],blocksize*thread,0,
								     sizesblocks[thread],T.dimensions[numhidlayers+1]);
    
  }



#pragma omp parallel for shared(blocklayerdata,blockreallayerdata,blockaux1,blockaux2)
  for(int thread=0;thread<maxnumthreads;++thread){
    for (int hid=0;hid<numhidlayers;++hid){

     
      //datahid[i-1]*wi

      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
		     1.0, &blocklayerdata[thread*(numhidlayers+2)+hid].matrix, &T.w[hid].matrix,
		     0.0, &blockreallayerdata[thread*(numhidlayers+1)+hid+1].matrix);


      //gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
      //	     1.0, T.layerdata[hid], &T.w[hid].matrix,
      //	     0.0, &T.reallayerdata[hid+1].matrix);

  
      //apply logistic function to the elements of reallayerdata[hid+1]

      logistic (&blockreallayerdata[thread*(numhidlayers+1)+hid+1].matrix,sizesblocks[thread],T.dimensions[hid+1]);

      //logistic (&T.reallayerdata[hid+1].matrix,T.batchsize,T.dimensions[hid+1]);


    }


    //last layer activation funstion is not identity

    //layerdata[numhidlayers]*w[numhidlayers]

     gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
 		   1.0, &blocklayerdata[thread*(numhidlayers+2)+numhidlayers].matrix, &T.w[numhidlayers].matrix,
 		   0.0, &blocklayerdata[thread*(numhidlayers+2)+numhidlayers+1].matrix);


     //gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
     //	   1.0, T.layerdata[numhidlayers], &T.w[numhidlayers].matrix,
     //	   0.0, &T.matrixlastlayer.matrix);

     logistic (&blocklayerdata[thread*(numhidlayers+2)+numhidlayers+1].matrix,sizesblocks[thread],T.dimensions[numhidlayers+1]);

     //logistic (&T.matrixlastlayer.matrix,T.batchsize,T.dimensions[numhidlayers+1]);


    gsl_matrix_memcpy (&blocklastlayerdata_matrix[thread].matrix,
		       &blocklayerdata[thread*(numhidlayers+2)+numhidlayers+1].matrix);    

    //gsl_matrix_memcpy (T.lastlayerdata_matrix,&T.matrixlastlayer.matrix);    

    
    //compute error matrix

    gsl_matrix_sub (&blocklayerdata[thread*(numhidlayers+2)+numhidlayers+1].matrix,
		    &blockoutputdata[thread].matrix);

    //gsl_matrix_sub (&T.matrixlastlayer.matrix,T.batchoutputdata);
    
    
    gsl_matrix_memcpy(&blockaux2[thread*(numhidlayers+1)+numhidlayers].matrix,
		      &blocklayerdata[thread*(numhidlayers+2)+numhidlayers+1].matrix);

    //T.aux2[numhidlayers]=&T.matrixlastlayer.matrix;



    gsl_matrix_mul_elements (&blockaux2[thread*(numhidlayers+1)+numhidlayers].matrix
			     ,&blocklastlayerdata_matrix[thread].matrix);

    //gsl_matrix_mul_elements (T.aux2[numhidlayers],T.lastlayerdata_matrix);



    gsl_matrix_scale (&blocklastlayerdata_matrix[thread].matrix, -1.0);

    //gsl_matrix_scale (T.lastlayerdata_matrix, -1.0);


    gsl_matrix_add_constant(&blocklastlayerdata_matrix[thread].matrix, 1.0);

    //gsl_matrix_add_constant(T.lastlayerdata_matrix, 1.0);


    gsl_matrix_mul_elements (&blockaux2[thread*(numhidlayers+1)+numhidlayers].matrix,
			     &blocklastlayerdata_matrix[thread].matrix);

    //gsl_matrix_mul_elements (T.aux2[numhidlayers],T.lastlayerdata_matrix);


    for(int hid=numhidlayers-1;hid>-1;--hid){

      gsl_matrix_memcpy (&blockaux1[thread*numhidlayers+hid].matrix,&blockreallayerdata[thread*(numhidlayers+1)+hid+1].matrix);   
      
      //gsl_matrix_memcpy (T.aux1[hid],&T.reallayerdata[hid+1].matrix);


      gsl_matrix_scale (&blockaux1[thread*numhidlayers+hid].matrix, -1.0);

      //gsl_matrix_scale (T.aux1[hid], -1.0);
   

      gsl_matrix_add_constant (&blockaux1[thread*numhidlayers+hid].matrix, 1.0);

      //gsl_matrix_add_constant (T.aux1[hid], 1.0);


      gsl_matrix_mul_elements (&blockaux1[thread*numhidlayers+hid].matrix,&blockreallayerdata[thread*(numhidlayers+1)+hid+1].matrix);

      //gsl_matrix_mul_elements (T.aux1[hid],&T.reallayerdata[hid+1].matrix);


      gsl_blas_dgemm(CblasNoTrans, CblasTrans,
		     1.0, &blockaux2[thread*(numhidlayers+1)+hid+1].matrix, &T.a[hid+1].matrix,
		     0.0, &blockaux2[thread*(numhidlayers+1)+hid].matrix);
   

      //gsl_blas_dgemm(CblasNoTrans, CblasTrans,
      //	     1.0, T.aux2[hid+1], &T.a[hid+1].matrix,
      //	     0.0, T.aux2[hid]);


      gsl_matrix_mul_elements (&blockaux2[thread*(numhidlayers+1)+hid].matrix,&blockaux1[thread*numhidlayers+hid].matrix);
   
      //gsl_matrix_mul_elements (T.aux2[hid],T.aux1[hid]);//result goes to aux2



    }

  }

  //update gradients outside the big cycle



  //gsl_blas_dgemm(CblasTrans, CblasNoTrans,
  //	 1.0, T.layerdata[numhidlayers], &T.aux2[numhidlayers].matrix,
  //	 0.0, &T.dw[numhidlayers].matrix);


  for(int hid=numhidlayers;hid>-1;--hid)

    gsl_blas_dgemm(CblasTrans, CblasNoTrans,
		   1.0, T.layerdata[hid], T.aux2[hid],
		   0.0, &T.dw[hid].matrix);


  //scalar error

  *error=gsl_blas_dnrm2 (T.lastlayer);									  



  //delete [] blockdw;

  delete [] blockaux1;

  delete [] blockaux2;

  delete [] blocklayerdata;

  delete [] blockreallayerdata;

  delete [] blockoutputdata;

  delete [] sizesblocks;

  delete []   blocklastlayerdata_matrix;



}



