     /* glue2mlps.cpp - create a new mlp putting side by side two mlps
        Copyright (C) 2010  Rui Rodrigues <rapr@fct.unl.pt>
        This software is released under the terms of the GNU General
        Public License (http://www.gnu.org/copyleft/gpl.html).
     */


#include <stdio.h>
#include <string>
#include <sys/time.h>
#include "mlp.h"
using namespace std;



mlp  glue2mlps(mlp&mlp1,mlp&mlp2){

  if(mlp1.numhidlayers!=mlp2.numhidlayers)
    throw "numhidlayers are not the same can not glue mlps!";


  unsigned numhidlayers=mlp1.numhidlayers;


  //dimlayers

  vector<unsigned> dimlayers(numhidlayers+2);

  for (unsigned i=0;i<numhidlayers+2;++i)
    dimlayers[i]=mlp1.dimlayers[i]+mlp2.dimlayers[i];


  //dimmatrices

  vector<unsigned> dimmatrices(2*(numhidlayers+1));

  for(unsigned i=0;i< numhidlayers+1;++i){

    dimmatrices[2*i]=dimlayers[i]+1;//includes bias

    dimmatrices[2*i+1]=dimlayers[i+1];
  }

  
  //mlpmatrix

  vector<gsl_matrix*> mlpmatrix(numhidlayers+1);

  for(unsigned i=0;i<numhidlayers+1;++i)
    mlpmatrix[i]=gsl_matrix_calloc(dimmatrices[2*i],dimmatrices[2*i+1]);

  
  //copy values from mlp1.mlpmatrix

  for(unsigned i=0;i<numhidlayers+1;++i){

    //first weights except bias

    gsl_matrix_view submatrix1=gsl_matrix_submatrix(mlpmatrix[i],0,0,mlp1.dimlayers[i],
						    mlp1.dimlayers[i+1]);

    gsl_matrix_const_view submatrix2=gsl_matrix_const_submatrix(mlp1.mlpmatrix[i],0,0,mlp1.dimlayers[i],
						    mlp1.dimlayers[i+1]);
    gsl_matrix_memcpy(&submatrix1.matrix,&submatrix2.matrix);


    //now bias

    gsl_matrix_view submatrix3=gsl_matrix_submatrix(mlpmatrix[i],mlp1.dimlayers[i]+mlp2.dimlayers[i],
						    0,1,mlp1.dimlayers[i+1]);

    gsl_matrix_const_view submatrix4=gsl_matrix_const_submatrix(mlp1.mlpmatrix[i],mlp1.dimlayers[i],0,
								1,mlp1.dimlayers[i+1]);

    gsl_matrix_memcpy(&submatrix3.matrix,&submatrix4.matrix);
  }


  
  //copy values from mlp2.mlpmatrix

  for(unsigned i=0;i<numhidlayers+1;++i){

    //first weights except bias

    gsl_matrix_view submatrix1=gsl_matrix_submatrix(mlpmatrix[i],mlp1.dimlayers[i],mlp1.dimlayers[i+1],
						    mlp2.dimlayers[i],mlp2.dimlayers[i+1]);

    gsl_matrix_const_view submatrix2=gsl_matrix_const_submatrix(mlp2.mlpmatrix[i],0,0,
								mlp2.dimlayers[i],
								mlp2.dimlayers[i+1]);

    gsl_matrix_memcpy(&submatrix1.matrix,&submatrix2.matrix);


    //now bias

    gsl_matrix_view submatrix3=gsl_matrix_submatrix(mlpmatrix[i],mlp1.dimlayers[i]+mlp2.dimlayers[i],
						    mlp1.dimlayers[i+1],1,mlp2.dimlayers[i+1]);

    gsl_matrix_const_view submatrix4=gsl_matrix_const_submatrix(mlp2.mlpmatrix[i],mlp2.dimlayers[i],0,
								1,mlp2.dimlayers[i+1]);

    gsl_matrix_memcpy(&submatrix3.matrix,&submatrix4.matrix);
  }

  mlp net(numhidlayers,dimlayers,mlpmatrix);

  return net;

}
