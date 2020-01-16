
     /* reconstructfrompatchesstartingfrom.cpp - given the patches with a certain delay('jump'),
	reconstruct the long signal
        Copyright (C) 2010  Rui Rodrigues <rapr@fct.unl.pt>
        This software is released under the terms of the GNU General
        Public License (http://www.gnu.org/copyleft/gpl.html).
     */




#include <stdio.h>
#include <string>
#include <algorithm>
#include <sys/time.h>

#include <iostream>

using namespace std;


#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

#ifdef _OPENMP
#include <omp.h>
#endif



void reconstructfrompatches_startingfrom(gsl_vector*longsignal, gsl_matrix*patches,
				    int start, int comp, int jump){

// #ifdef _OPENMP
// const int maxnumthreads = omp_get_max_threads();
// #else /* _OPENMP */
// const int maxnumthreads = 1;
// #endif /* _OPENMP */



  int patchsize=(int) patches->size2;

  int numpatches=(int) patches->size1;


  if((numpatches-1)*jump+patchsize<comp-start-1){

    cout<<"numpatches, jump, patchsize,start and comp are not compatible!"<<endl;
    cout<<"numpatches= "<<numpatches<<"  jump= "<<jump<<"  patchsize "<<patchsize<<"  start= "<<start<<" comp= "<<comp<<endl;

    exit(1);
  }

  if((int) longsignal->size<comp){

    cout<<"lonsignal->size is smaller than comp!"<<endl;
    exit(1);
  }

  gsl_vector_set_zero(longsignal);



  int aux, last, first;

  int numsegmentscoverpoint=patchsize/jump;

  int firstbigger=start+comp;

#pragma omp parallel for private(last,aux,first)
  for(int i=start;i<firstbigger;++i){

    
    last=min(i/jump,(int)numpatches-1);

    aux=min(i/jump-numsegmentscoverpoint+1,(int)numpatches-1);

    first=max( 0,aux);

    
    for(int k=first;k<=last;++k)
      gsl_vector_set(longsignal,i-start,
		     gsl_vector_get(longsignal,i-start)+
		     gsl_matrix_get(patches,k,i-k*jump));

    gsl_vector_set(longsignal,i-start,gsl_vector_get(longsignal,i-start)/(last-first+1));
      
  }

}
