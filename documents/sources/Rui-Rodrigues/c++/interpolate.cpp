     /* interpolate.cpp - use GSL interpolation to interpolate outputdata
        Copyright (C) 2010  Rui Rodrigues <rapr@fct.unl.pt>
        This software is released under the terms of the GNU General
        Public License (http://www.gnu.org/copyleft/gpl.html).
     */


#include <gsl/gsl_matrix.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>

#include <iostream>
#include <fstream>
#include <vector>
using namespace std;



void interpolatepatchdata(gsl_matrix*&patchdata,int newpatchsize,int step){


  int npatches=patchdata->size1;

  int patchsize=patchdata->size2;



  gsl_matrix *newpatchdata=gsl_matrix_alloc(npatches,newpatchsize);



  double x[patchsize];



  for(int k=0, i=0;k<patchsize;++k,i+=step)
      x[k]=(double)i;

  double*y;


  for(int i=0;i<npatches;++i){

    //y should point to the right patch

    y=patchdata->data+i*patchsize;

    
    gsl_interp_accel *acc= gsl_interp_accel_alloc ();

    gsl_spline *spline= gsl_spline_alloc (gsl_interp_cspline, patchsize);

    gsl_spline_init (spline, x, y, patchsize);

    
    for (int j = 0; j <newpatchsize; ++j)

      gsl_matrix_set(newpatchdata,i,j, gsl_spline_eval (spline, j, acc));


    gsl_spline_free (spline);
    
    gsl_interp_accel_free (acc);
  }


  gsl_matrix *aux;


  aux=patchdata;

  patchdata=newpatchdata;

  newpatchdata=aux;


  gsl_matrix_free(newpatchdata);
}
