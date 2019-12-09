     /* gradgsl.h definition of classes tu use GSL optmization in backprop 
        Copyright (C) 2010  Rui Rodrigues <rapr@fct.unl.pt>
        This software is released under the terms of the GNU General
        Public License (http://www.gnu.org/copyleft/gpl.html).
     */

#include <string>

#include <string.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_blas.h>



class parametersfwdgsl{
 public:


  parametersfwdgsl(int Numhiddenlayers,int*Dimensions,int Batchsize);

  ~parametersfwdgsl();

  //gsl_matrix* batchoutputdata;

  int numhidlayers;

  int * dimensions;

  int batchsize;

  int ninputs;

  //total number of weights+bias in each layer
  int * tam;

  int tamtotal;

  //inicio da matriz w[i] no vector que contém todos os pesos e bias
  int *initw;//[numhidlayers+1];

 
  gsl_matrix ** layerdata;


  gsl_matrix_view *reallayerdata;

  gsl_matrix_view *w;


  gsl_matrix * fwd_data; 
};



 

class parametersgradgsl{

 public:


  parametersgradgsl(int Numhiddenlayers,int*Dimensions,int Batchsize);

  ~parametersgradgsl();

  gsl_matrix* batchoutputdata;

  int numhidlayers;

  int * dimensions;

  int batchsize;

  int ninputs;

  int *nhidden;//[numhidlayers+1];//nhidden[numhidlayers]=noutputs

  //total number of weights+bias in each layer
  int * tam;

  int tamtotal;

  //inicio da matriz w[i] no vector que contém todos os pesos e bias
  int *initw;//[numhidlayers+1];
 
  gsl_matrix ** layerdata;

  gsl_vector * lastlayer;

  gsl_matrix * lastlayerdata_matrix;

  gsl_matrix_view matrixlastlayer;


  gsl_matrix_view *reallayerdata;


  gsl_matrix_view *w;


  gsl_matrix ** aux1, ** aux2;


  gsl_matrix_view * dw;

  gsl_matrix_view *a;
    
};



//here td is parametersfwdgsl*

void fwdgsl_vislinear(const gsl_vector * x,void * td,gsl_matrix*);

double just_compute_error_gsl_vislinear(const gsl_vector * x,
					  void * td,gsl_matrix*batchoutputdata); 

void give_error_for_each_patch_gsl_vislinear(const gsl_vector * x,void * td,gsl_matrix*batchoutputdata,
					     double*errorforpatch);



//here td is parametersgradgsl*

double compute_error_gsl_vislinear(const gsl_vector * x,void * td);

void gradientgsl_vislinear(const gsl_vector* g,void*parameters,gsl_vector *grad);

void errorandgrad_vislinear(const gsl_vector * x, void * params, double * error, gsl_vector * grad);


//here td is parametersfwdgsl*

void fwdgsl_vislogistic(const gsl_vector * x,void * td,gsl_matrix*);

double just_compute_error_gsl_vislogistic(const gsl_vector * x,
					  void * td,gsl_matrix*batchoutputdata); 


//here td is parametersgradgsl*

double compute_error_gsl_vislogistic(const gsl_vector * x,void * td);

void  gradientgsl_vislogistic(const gsl_vector* x,void*td, gsl_vector *grad);

void  errorandgrad_vislogistic(const gsl_vector* x,void*td, double * error, gsl_vector *grad); 
