

     /* mlp.cpp - implement definitions of mlp.h
        Copyright (C) 2010  Rui Rodrigues <rapr@fct.unl.pt>
        This software is released under the terms of the GNU General
        Public License (http://www.gnu.org/copyleft/gpl.html).
     */


#include "mlp.h"

#include <iostream>

mlp::mlp(unsigned Numhidlayers, vector<unsigned>& Dimlayers){


  if (Numhidlayers!=Dimlayers.size()-2)

    throw (const char*) "numlayers and dimlayers don t agree on two argument constructor!";


  numhidlayers=Numhidlayers;

  dimlayers=Dimlayers;


  //buid dimmatrices

  dimmatrices=vector<unsigned>(2*(numhidlayers+1));

  for(unsigned i=0;i< numhidlayers+1;++i){

    dimmatrices[2*i]=dimlayers[i]+1;//includes bias

    dimmatrices[2*i+1]=dimlayers[i+1];
  }
 

  //build mlpmatrix

  mlpmatrix=vector<gsl_matrix*>(numhidlayers+1);

  for(unsigned i=0;i<numhidlayers+1;++i)
    mlpmatrix[i]=gsl_matrix_calloc(dimmatrices[2*i],dimmatrices[2*i+1]);
}





mlp::mlp(unsigned Numhidlayers, vector<unsigned>& Dimlayers,vector<gsl_matrix*>& Mlpmatrix){

  init(Numhidlayers,Dimlayers,Mlpmatrix);
}


mlp::mlp(const mlp& mlp_other){

init(mlp_other.numhidlayers,mlp_other.dimlayers,mlp_other.mlpmatrix);
}


mlp::~mlp(){

  for(unsigned i=0;i<numhidlayers+1;++i)

    gsl_matrix_free(mlpmatrix[i]);

}


void mlp::operator=(const mlp&mlp_other){

  if(this!=&mlp_other){

    for(unsigned i=0;i<numhidlayers+1;++i)
      gsl_matrix_free(mlpmatrix[i]);


    init(mlp_other.numhidlayers,mlp_other.dimlayers,mlp_other.mlpmatrix);
  }
    
}



void mlp::getweightsfromgslvector(gsl_vector* container){


  //check dimensions
 
  unsigned numweights=0;

  for(unsigned i=0;i<numhidlayers+1;++i)
    numweights+=dimmatrices[2*i]*dimmatrices[2*i+1];

  if(numweights!=container->size)

    throw (const char*) "mlp-matrices dimensions and vector size are not compatible!";

  else{

    unsigned start=0;

    for(unsigned i=0;i<numhidlayers+1;++i){


      gsl_vector_view subvector=gsl_vector_subvector (container,start,numweights-start);

      gsl_matrix_memcpy (mlpmatrix[i],
			   &gsl_matrix_view_vector (&subvector.vector,
						    dimmatrices[2*i],
						    dimmatrices[2*i+1]).matrix);

      start+=dimmatrices[2*i]*dimmatrices[2*i+1];

    }
  }

}




//can read backprop weights

void mlp::loadtxtweights(const char* filename){


  FILE* readfromfile=fopen(filename,"r");


  unsigned num_matrices;

  fscanf(readfromfile,"%u",&num_matrices);

  if(num_matrices!=numhidlayers+1)

    throw (const char*) "num matrices on file differs from mlp num matrices!";


  vector<unsigned> nrows(num_matrices), ncols(num_matrices);


  for(unsigned i=0;i<numhidlayers+1;++i){

    fscanf(readfromfile,"%u",&(nrows[i]));

    fscanf(readfromfile,"%u",&(ncols[i]));

    if((nrows[i]!=dimmatrices[2*i])||(ncols[i]!=dimmatrices[2*i+1]))
      throw (const char*) "dimensions of matrix in file diferent from mlp matrix"; 
  }

  bool sucess=true;  

  int res=0;

  for(unsigned i=0;i<numhidlayers+1;++i){ 

    res=gsl_matrix_fscanf(readfromfile,mlpmatrix[i]);

    if(res!=0) sucess=false;
  }

  if(!sucess)
    throw "problem reading gsl_matrix from file";

  fclose(readfromfile);
}





//saces like backprop weights


void mlp::savetxtweights( const char* filename){

  FILE* writeonfile=fopen(filename,"w");

  fprintf(writeonfile,"%u ",numhidlayers+1);

  for(unsigned i=0;i<numhidlayers+1;++i)
        fprintf(writeonfile,"%u %u ", dimmatrices[2*i], dimmatrices[2*i+1]);

  fprintf(writeonfile,"\n");

  for(unsigned i=0;i<numhidlayers+1;++i)
    gsl_matrix_fprintf(writeonfile,mlpmatrix[i],"%11.8g");

  fclose(writeonfile);
}


void checkinitialdatacompatibility( const unsigned Numhidlayers, const vector<unsigned>& Dimlayers,

				    const vector<gsl_matrix*>& mlpmatrix){


  if (Numhidlayers!=Dimlayers.size()-2)

    throw (const char*) "numlayers and dimlayers don t agree!";

  
  unsigned nrows,ncols;

  for(unsigned i=0;i<Numhidlayers+1;++i){
  
    nrows=mlpmatrix[i]->size1;

    ncols=mlpmatrix[i]->size2;

    if((nrows!=Dimlayers[i]+1)||(ncols!=Dimlayers[i+1]))

      throw (const char*) "dimlayers and mlpmatrix dimensions don t agre don t agree!";
  }
  
}



void mlp::init(const unsigned Numhidlayers,const vector<unsigned>& Dimlayers, const vector<gsl_matrix*>& Mlpmatrix){

   
  checkinitialdatacompatibility( Numhidlayers, Dimlayers, Mlpmatrix); 


  numhidlayers=Numhidlayers;


  dimlayers=Dimlayers;


  //buid dimmatrices

  dimmatrices=vector<unsigned>(2*(numhidlayers+1));

  for(unsigned i=0;i< numhidlayers+1;++i){

    dimmatrices[2*i]=dimlayers[i]+1;//includes bias

    dimmatrices[2*i+1]=dimlayers[i+1];
  }
 


  //build mlpmatrix coping data from Mlpmatrix

  mlpmatrix=vector<gsl_matrix*>(numhidlayers+1);

  for(unsigned i=0;i<numhidlayers+1;++i){

    mlpmatrix[i]=gsl_matrix_alloc(dimmatrices[2*i],dimmatrices[2*i+1]);

    gsl_matrix_memcpy (mlpmatrix[i],Mlpmatrix[i]);
    
  }

}
