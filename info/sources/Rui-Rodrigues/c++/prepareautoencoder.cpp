

     /* prepareautoencoder.cpp - load the coeficients from rbm vislinear and rbm logistic 
	and organize them as the four matrices of an
	autoencoder and save them on the (right) file
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


#include "netdimsandfilenames.h"
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>


void checkfstream(ofstream& file_io,const char* filename);
//in checkfstream.cpp

void checkfstream(ifstream& file_io,const char* filename);
//in checkfstream.cpp


void read_datafile(ifstream&in,netdimsandfilenames& A);
//in netdimsandfilenames.cpp

void writegslmatriz(const char* filename,gsl_matrix*m);
void readgslmatriz(const char* filename,gsl_matrix*&m);
void readgslvectorasmatriz(const char* filename,gsl_vector*v);
void readgslvector(const char* filename,gsl_vector*m);
//in iogslvectormatrix.cpp



int main(int argc, char ** argv){


  try{

if(argc<3){

    cout<<" must be called with arguments  signal1  and folder name !"<<endl;

    exit(0);
  }



 //get filename to extract dimensions and else--


  //signal1

  string aa1=argv[1];
  string cc=argv[2];
  string b1=aa1.append(".txt");
  string d1="../";
  d1.append(cc);
  d1.append("/");
  d1.append(b1);

  cout<<"reading configuration data from "<<d1.c_str()<<endl;

  netdimsandfilenames A1;

  ifstream reading1(d1.c_str());

  read_datafile(reading1,A1);

  checkfstream(reading1,d1.c_str());

  reading1.close();



  unsigned ninputs=A1.patchsize*A1.nsignals, nhidden0=A1.nhidden0, nhidden1=A1.nhidden1;
  

  //load weights from rbms

  gsl_vector*weightsandbiasrbmvislinear=gsl_vector_alloc(ninputs*nhidden0+ninputs+nhidden0);

  readgslvector(A1.netvislinearweights.c_str(),weightsandbiasrbmvislinear);


  gsl_vector*weightsandbiasrbmlogistic=gsl_vector_alloc(nhidden0*nhidden1+nhidden0+nhidden1);

  readgslvector(A1.netlogisticweights.c_str(),weightsandbiasrbmlogistic);



  gsl_matrix* m1,*m2,*m3,*m4;


  m1=gsl_matrix_alloc(ninputs+1,nhidden0);
  
  m2=gsl_matrix_alloc(nhidden0+1,nhidden1);

  m3=gsl_matrix_alloc(nhidden1+1,nhidden0);

  m4=gsl_matrix_alloc(nhidden0+1,ninputs);



  //m1, m4

  gsl_matrix_view aux1=gsl_matrix_view_vector(weightsandbiasrbmvislinear,ninputs,nhidden0);

  gsl_matrix_view aux2=gsl_matrix_submatrix(m1,0,0,ninputs,nhidden0);

  gsl_matrix_memcpy(&aux2.matrix,&aux1.matrix);


  gsl_vector_view aux3=gsl_vector_subvector(weightsandbiasrbmvislinear,ninputs*nhidden0+ninputs,nhidden0);

  gsl_matrix_set_row (m1, ninputs, &aux3.vector);


  gsl_matrix_view aux4=gsl_matrix_submatrix(m4,0,0, nhidden0,ninputs);

  gsl_matrix_transpose_memcpy (&aux4.matrix, &aux2.matrix);


  gsl_vector_view aux5=gsl_vector_subvector(weightsandbiasrbmvislinear,ninputs*nhidden0,ninputs);

  gsl_matrix_set_row (m4, nhidden0, &aux5.vector);



  //m2, m3

  aux1=gsl_matrix_view_vector(weightsandbiasrbmlogistic,nhidden0,nhidden1);

  aux2=gsl_matrix_submatrix(m2,0,0,nhidden0,nhidden1);

  gsl_matrix_memcpy(&aux2.matrix,&aux1.matrix);


  aux3=gsl_vector_subvector(weightsandbiasrbmlogistic,nhidden0*nhidden1+nhidden0,nhidden1);

  gsl_matrix_set_row (m2, nhidden0, &aux3.vector);


  aux4=gsl_matrix_submatrix(m3,0,0,nhidden1,nhidden0);

  gsl_matrix_transpose_memcpy (&aux4.matrix, &aux2.matrix);


  aux5=gsl_vector_subvector(weightsandbiasrbmlogistic,nhidden0*nhidden1,nhidden0);

  gsl_matrix_set_row (m3, nhidden1, &aux5.vector);




  //save coefficients autoencoder on file
  

  FILE*autoencodercoefs=fopen(A1.backpropautoencodercoefficientsfile.c_str(),"w");

  fprintf(autoencodercoefs, "%u %u %u %u %u %u %u %u %u \n", 4, ninputs+1, nhidden0, nhidden0+1, nhidden1, nhidden1+1, nhidden0, nhidden0+1,ninputs);

  gsl_matrix_fprintf(autoencodercoefs,m1,"%lf");

  gsl_matrix_fprintf(autoencodercoefs,m2,"%lf");

  gsl_matrix_fprintf(autoencodercoefs,m3,"%lf");

  gsl_matrix_fprintf(autoencodercoefs,m4,"%lf");

  fclose(autoencodercoefs);


  //free

  gsl_vector_free(weightsandbiasrbmvislinear);

  gsl_vector_free(weightsandbiasrbmlogistic);

  gsl_matrix_free(m1);

  gsl_matrix_free(m2);

  gsl_matrix_free(m3);

  gsl_matrix_free(m4);

  }

  catch(int i){

    cout<<" caught exception "<<i<<endl;

    exit(1);
  }

  return 0;
}
