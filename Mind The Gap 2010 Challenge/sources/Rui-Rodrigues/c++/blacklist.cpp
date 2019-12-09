#include <stdio.h>
#include <string>
#include <sys/time.h>

using namespace std;

#include <algorithm>
#include<vector>
#include <iostream>
#include <fstream>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "netdimsandfilenames.h"
#include <set>

void useblacklist(gsl_matrix * &inputdata,const char*blacklistfile){


  set<int> blacklist;

  ifstream reading(blacklistfile);

  cout<<"loading blacklist!"<<endl;

  int cont=0;

  if(!reading){

    cout<<"can not open file"<<endl;

    exit(1);
  }

  int aux=-1;

  while (!reading.eof( )){

    reading>>aux;

    if(aux>-1){

      blacklist.insert(aux);
    
      printf("%d \r",cont);

      ++cont;
    }
  }  

  cout<<endl;

  int sizeblacklist=blacklist.size();

  int initialnpatches=inputdata->size1;

  int newnpatches=initialnpatches-sizeblacklist;


  gsl_matrix*newinputdata=gsl_matrix_alloc(newnpatches,inputdata->size2);

  set<int>::iterator it;

  gsl_vector_view vecaux;

  int j=0;

  for(int i=0;i<initialnpatches;++i){

    it=blacklist.find(i);

    if(it==blacklist.end()){

      vecaux=gsl_matrix_row (inputdata,i);

      gsl_matrix_set_row (newinputdata, j, &vecaux.vector); 

      ++j;
    }
  }

  gsl_matrix*temp;

  temp=inputdata;

  inputdata=newinputdata;

  newinputdata=temp;

  gsl_matrix_free(newinputdata);
}


