     /* create_autoencoderblacklist.cpp - list the training samples where autoencoder
	error is much worse than in test set
        Copyright (C) 2010  Rui Rodrigues <rapr@fct.unl.pt>
        This software is released under the terms of the GNU General
        Public License (http://www.gnu.org/copyleft/gpl.html).
     */


#include <stdio.h>
#include <string>
#include <sys/time.h>

using namespace std;

#include <algorithm>
#include<vector>
#include<set>
#include <iostream>
#include <fstream>
#include "gradgsl.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "netdimsandfilenames.h"



void checkfstream(ofstream& file_io,const char* filename);
//in ../geral/checkfstream.cpp

void checkfstream(ifstream& file_io,const char* filename);
//in ../geral/checkfstream.cpp

void checkfstream(FILE* pointer,const char* filename);



void read_datafile(ifstream&in,netdimsandfilenames& A);
//in netdimsandfilenames.cpp


void readgslvectormatrix(FILE* readfromfile, gsl_vector* v,vector<int>&sizes);
//in iosgslvectormatrix.cpp

void readgslmatriz(const char* filename,gsl_matrix*&m);
//in iosgslvectormatrix.cpp

void writegslmatriz(const char* filename,gsl_matrix*m);
//save a vector that contains several gslmatrices 
void savegslvectormatrix(FILE* writetofile, gsl_vector* v,vector<int>&sizes);
//in iosgslvectormatrix.cpp


void readconfigurationdata(string&file,  netdimsandfilenames&A);
//down in this file

int give_autoencoder_weights_sizes(netdimsandfilenames&A);
//down in this file

void loadautoencoderweights(netdimsandfilenames&A, gsl_vector*weights);


void useblacklist(gsl_matrix * &inputdata,const char*blacklistfile);
//in blacklist.cpp

//the following will be use to choose to load initial weights for rbmlogistic or backproprbm weights

const int num_matrices_autoencoder=4;




// -------------------------------------------------------------------



// ----------------------------------------------------------------------------------------

int main(int argc, char ** argv){


  try{

if(argc<3){

    cout<<" must be called with argument  signal  and folder name !"<<endl;

    exit(0);
  }





  //time -----------------
  //struct timeval start, end;
  //gettimeofday(&start, NULL);


  //get filenames to extract dimensions and else--

  //folder
  string folder="../";
  folder.append(argv[2]);
  folder.append("/");

  //signal
  string signal=argv[1];
  string d=folder;
  d.append(signal);
  d.append(".txt");

  cout<<"reading configuration data from "<<d.c_str()<<endl;
  netdimsandfilenames A;
  readconfigurationdata(d, A);




  //-------------change in case of different number of layers---------------
  int nhidden0=A.nhidden0, nhidden1=A.nhidden1, 
    nhidden2=A.nhidden0;
  int ninputs=A.nsignals*A.patchsize;
  int dimensions[]={ninputs,nhidden0,nhidden1,nhidden2,ninputs};

 
  //size_t numhidlayers=3;
  //----------------------------------------------------
  //load initial training data

 
  gsl_matrix * inputdata;

  cout<<"loading inputdata!"<<endl;

  readgslmatriz(A.patchdatafile.c_str(),inputdata);

  cout<<"at the begining numpatches is "<<inputdata->size1<<endl;


  if((int) inputdata->size2!=ninputs){
    cout<<"inputdata is not compatible with ninputs!"<<endl;
    exit(1);
  }


  //load autoencoder weights

  gsl_vector * autoencoderweights=gsl_vector_alloc (give_autoencoder_weights_sizes(A));

  loadautoencoderweights(A,autoencoderweights);


  
  //load signal patches on critical time

  string criticaltime=folder;

  criticaltime.append("criticaltime_");

  criticaltime.append(signal);

  criticaltime.append(".txt");


  gsl_matrix * criticaltimedata;

  readgslmatriz(criticaltime.c_str(),criticaltimedata);


  //compute features from critical time

  int nsegments=criticaltimedata->size1;



  //compute autoencoder error by patch in the last 30 secs

  double *errorforpatchcritical_time=new double[nsegments];

  double maximum_errorforpatchcritical_time;

  double other_error_rate_critical_time;

  double *diferenceoncritical_time=new double[nsegments];

  double minimumdiferenceoncriticaltime, max,min;

  gsl_vector_view vecaux;

  {
    parametersfwdgsl parameters(3,dimensions,nsegments); 

    gsl_matrix_memcpy(&(parameters.reallayerdata[0].matrix),criticaltimedata);

    //error for each patch
    give_error_for_each_patch_gsl_vislinear(autoencoderweights,&parameters,
					    criticaltimedata,errorforpatchcritical_time);

    //maximum from error for each patch

    vecaux=gsl_vector_view_array (errorforpatchcritical_time,nsegments);

    maximum_errorforpatchcritical_time=gsl_vector_max (&vecaux.vector);


  
    other_error_rate_critical_time=just_compute_error_gsl_vislinear(autoencoderweights,&parameters,
								    criticaltimedata);


    //diference between max and min on each patch

    for(int i=0;i<nsegments;++i){

      vecaux=gsl_matrix_row (criticaltimedata,i);

      gsl_vector_minmax (&vecaux.vector, &min, &max);

      diferenceoncritical_time[i]=max-min;
    }
 
   
   //min difference

    vecaux=gsl_vector_view_array (diferenceoncritical_time,nsegments);
    minimumdiferenceoncriticaltime=gsl_vector_min (&vecaux.vector);



    cout<<"error rate on  in the last segments is "<<other_error_rate_critical_time*other_error_rate_critical_time/nsegments<<endl;

    cout<<"max error by patch on  the last segments is "<<maximum_errorforpatchcritical_time*maximum_errorforpatchcritical_time<<endl;

    cout<<"minimum diference on patch during critical time is "<< minimumdiferenceoncriticaltime<<endl;

  }


  //now compute error of  autoencoder on training time for each patch

  int npatches=inputdata->size1;

  double *errorforpatchtraining=new double[npatches];

  double *diferenceforpatchtraining=new double[npatches];


  parametersfwdgsl parameters(3,dimensions,npatches); 

  gsl_matrix_memcpy(&(parameters.reallayerdata[0].matrix),inputdata);
  

  give_error_for_each_patch_gsl_vislinear(autoencoderweights,&parameters,
					  inputdata,errorforpatchtraining);
  


  //compute 'diference' for each patch

  

  for(int i=0;i<npatches;++i){
 
    vecaux=gsl_matrix_row (inputdata,i);

    gsl_vector_minmax (&vecaux.vector, &min, &max);

    diferenceforpatchtraining[i]=max-min;

  }


  //build blacklist

  set<int> blacklist;

  double factorformaximumerror=4.0;

  double factorfordiference=1.0/10;


  for(int i=0;i<npatches;++i)
    if((errorforpatchtraining[i]>factorformaximumerror*maximum_errorforpatchcritical_time)||
       (diferenceforpatchtraining[i]<factorfordiference*minimumdiferenceoncriticaltime))
      blacklist.insert(i);


  cout<<"blacklist size is "<<blacklist.size()<<endl;


  //put blacklist on file

  string blacklistfile=folder;
  blacklistfile.append(signal);
  blacklistfile.append("_blacklist.txt");


  cout<<"saving blacklist on file: "<<blacklistfile<<endl;

  ofstream saveonfile(blacklistfile.c_str(), ios::out);

  for( set<int>::const_iterator iter = blacklist.begin();
       iter != blacklist.end();
       ++iter ) {

    saveonfile<<*iter << '\n';
  }

  saveonfile<<endl;




  }
  catch(int a){}

  return 0;

  }


int give_autoencoder_weights_sizes(netdimsandfilenames&A){

  int ninputs=A.nsignals*A.patchsize;

  int retvalue=(ninputs+1)*(A.nhidden0)+(A.nhidden0+1)*(A.nhidden1)+(A.nhidden1+1)*A.nhidden0+(A.nhidden0+1)*ninputs;

  return retvalue;

}



void loadautoencoderweights(netdimsandfilenames&A, gsl_vector*weights){

  vector<int> knownsizes(9);

  int ninputs=A.nsignals*A.patchsize;

  knownsizes[0]=4;

  knownsizes[1]=ninputs+1;

  knownsizes[2]=A.nhidden0;

  knownsizes[3]=A.nhidden0+1;

  knownsizes[4]=A.nhidden1;

  knownsizes[5]=A.nhidden1+1;

  knownsizes[6]=A.nhidden0;

  knownsizes[7]=A.nhidden0+1;

  knownsizes[8]=ninputs;


  vector<int> sizes;

  FILE * readweights = fopen(A.backpropautoencodercoefficientsfile.c_str(), "r");

  readgslvectormatrix(readweights, weights,sizes);

  fclose(readweights);

  for(int i=0; i<9;++i)

  if(knownsizes[i]!=(int)sizes[i]){

    cout<<"matrices sizes are not correct!"<<endl;

    exit(1);
  }
}


void readconfigurationdata(string&file,  netdimsandfilenames&A){


 ifstream reading(file.c_str());

  read_datafile(reading,A);

  checkfstream(reading,file.c_str());

  reading.close();

}
