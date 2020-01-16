
     /* iogslvectormatrix.cpp - input and output to a file one or several matrices
        Copyright (C) 2010  Rui Rodrigues <rapr@fct.unl.pt>
        This software is released under the terms of the GNU General
        Public License (http://www.gnu.org/copyleft/gpl.html).
     */




#include <iostream>
#include <fstream>
#include <vector>
using namespace std;

#include <sys/time.h>

#include <string>
#include<vector>
#include <string.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>


using namespace std;


//save a vector that contains several gslmatrices 
void savegslvectormatrix(FILE* writetofile, gsl_vector* v,vector<int>&sizes){

  //int num_matrices=sizes[0];//sizes.size()=1+2*sizes[0];

  for(int i=0;i<1+2*sizes[0];++i)
    fprintf(writetofile,"%u ",sizes[i]);

  fprintf(writetofile,"\n");

  int failure=gsl_vector_fprintf (writetofile, v, "%11.8g");

  if(failure){

    cout<<"couldn t save gsl_vector!"<<endl; 
    exit(1);
  }
}


//read a vector that contains several gslmatrices 
void readgslvectormatrix(FILE* readfromfile, gsl_vector* v,vector<int>&sizes){

  if(readfromfile==0){

    cout<<"could not open file for reading"<<endl;

    throw "didn't open file for reading";
  }

  int num_matrices;

  if(fscanf(readfromfile,"%u",&num_matrices)!=1) exit(1);

  sizes.resize(1+2*num_matrices);

  sizes[0]=num_matrices;

  
  for(int i=1;i<1+2*num_matrices;++i)
    if(fscanf(readfromfile,"%u",&sizes[i])!=1) exit(1);


  int tamtotal=0;
  for(int i=0;i<num_matrices;++i)
    tamtotal+=sizes[2*i+1]*sizes[2*i+2];


  if((int) v->size!=tamtotal){
    cout<<"gsl_vector doesn t have theexpected size!"<<endl;
    exit(1);
  }

  int failure=gsl_vector_fscanf(readfromfile, v);

  if(failure){

    cout<<"couldn t read gsl_vector!"<<endl; 
    exit(1);
  }
}

//read a vector that contains several gslmatrices 
int givesize_gslvector_infile(FILE* readfromfile){  

  int num_matrices;

  if(fscanf(readfromfile,"%u",&num_matrices)!=1){

    cout<<"problem reading num_matrices"<<endl;

    exit(1);
  }

  vector<int> sizes(2*num_matrices);
 
  for(int i=0;i<2*num_matrices;++i)
    if(fscanf(readfromfile,"%u",&sizes[i])!=-1){

    cout<<"problem reading size matrices"<<endl;

    exit(1);
  } 


  int tamtotal=0;
  for(int i=0;i<num_matrices;++i)
    tamtotal+=sizes[2*i]*sizes[2*i+1];

  return tamtotal;

}


//this version receives a pointer to an alredy allocated gsl_matrix 

void readgslmatriz(const char* filename,gsl_matrix*m,int NLIN,int NCOL){

  FILE * readmatrix = fopen(filename,"r");

  int nlin, ncol;

  if(fscanf (readmatrix,"%u",&nlin)!=1){

    cout<<"problem reading nlin in readgslmatriz"<<endl;

    exit(1);
  } 

  if(fscanf (readmatrix,"%u",&ncol)!=1){

    cout<<"problem reading ncol"<<endl;

    exit(1);
  } 

  if((nlin!=NLIN)||(ncol!=NCOL)){

    throw "matrices dimensions don t fit!";
  }

  int a=gsl_matrix_fscanf(readmatrix,m);

  if(a!=0)
    throw "problem reading gsl_matrix!";

  fclose(readmatrix);
  
}


//reads into a vector the data of a matrix, vector must be alredy alocated

void readgslvectorasmatriz(const char* filename,gsl_vector*v){

  
  FILE * readmatrix = fopen(filename,"r");

  int nlin, ncol;

  if(fscanf (readmatrix,"%u",&nlin)!=1){

    cout<<"problem reading nlin in readgslvectorasmatriz"<<endl;

    exit(1);
  } 

  if(fscanf (readmatrix,"%u",&ncol)!=1){

    cout<<"problem reading ncol"<<endl;

    exit(1);
  } 

  if(nlin*ncol!=(int) v->size){

    throw "vector size doesn t fit with matrix in file!";
  }

  int a=gsl_vector_fscanf(readmatrix,v);

  if(a!=0)
    throw "problem reading gsl_vector!";

  fclose(readmatrix);
}


//this version receives a pointer to a gsl_matrix but it allocattes that gsl_matrix

void readgslmatriz(const char* filename,gsl_matrix*&m){

  FILE * readmatrix = fopen(filename,"r");

  int nlin, ncol;

  if(fscanf (readmatrix,"%u",&nlin)!=1){

    cout<<"problem reading nlin in readgslmatriz(version allocating memory)"<<endl;

    exit(1);
  } 

  if(fscanf (readmatrix,"%u",&ncol)!=1){

    cout<<"problem reading ncol"<<endl;

    exit(1);
  } 

  m=gsl_matrix_alloc(nlin,ncol);

  int a=gsl_matrix_fscanf(readmatrix,m);

  if(a!=0)
    throw "problem reading gsl_matrix!";

  fclose(readmatrix);
  
}


void writegslmatriz(const char* filename,gsl_matrix*m){

  FILE * writematrix = fopen(filename,"w");

  fprintf(writematrix,"%lu %lu\n",(int long) m->size1, (int long)m->size2);
  
  gsl_matrix_fprintf (writematrix,m,"%.8f");

  fclose(writematrix);
}



//used for rbm

void writegslvector(const char* filename,gsl_vector*m){

  FILE * writevector = fopen(filename,"w");

  fprintf(writevector,"%lu \n",(int long) m->size);

  gsl_vector_fprintf (writevector,m,"%.8f");

  fclose(writevector);
}


//this version receives a pointer to an alredy allocated gsl_vector 

void readgslvector(const char* filename,gsl_vector*m){

  FILE * readvector = fopen(filename,"r");

  int  nelements;

  if(fscanf (readvector,"%u",&nelements)!=1){

    cout<<"problem reading nelements"<<endl;

    exit(1);
  } 

 

  if(nelements!=(int) m->size){

    throw "vector dimensions don t fit!";
  }

  int a=gsl_vector_fscanf(readvector,m);

  if(a!=0)
    throw "problem reading gsl_vector!";

  fclose(readvector);
  
}
