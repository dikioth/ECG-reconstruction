
     /* checkfstream.cpp - chec if stream was correctely open
        Copyright (C) 2010  Rui Rodrigues <rapr@fct.unl.pt>
        This software is released under the terms of the GNU General
        Public License (http://www.gnu.org/copyleft/gpl.html).
     */




#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>

using namespace std;

void checkfstream(ifstream& file_io,const char* filename){

  if(!file_io){

    cerr<<filename<<' '<<" wasn't correctely read!"<<endl;

    exit(1);

  }

} 

void checkfstream(ofstream& file_io,const char* filename){

  if(!file_io){

   cerr<<filename<<' '<<" wasn't correctely written!"<<endl;

    exit(1);

  }

} 


void checkfstream(FILE* pointer,const char* filename){

  if(pointer==0){

    cerr<<filename<<' '<<" wasn't correctely open!"<<endl;

    exit(1);

  }

} 

