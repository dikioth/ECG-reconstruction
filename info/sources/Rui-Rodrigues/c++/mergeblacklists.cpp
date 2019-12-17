
     /* mergeblacklists.cpp 
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
#include <iostream>
#include <fstream>
#include "netdimsandfilenames.h"
#include <set>


int main(int argc, char ** argv){


  if(argc<5){

    cout<<" must be called first with signals names  then  newsignal name and finally folder name !"<<endl;
    exit(0);
  }


  string folder;
  folder="../";
  folder.append(argv[argc-1]);
  folder.append("/");


  set<int> newblacklist;

  string signal;

  for(int i=1;i<argc-2;++i){

    signal=argv[i];

    string readingfile=folder;
    readingfile.append(signal);  
    readingfile.append("_blacklist.txt");

    ifstream reading(readingfile.c_str());

    cout<<"reading blacklistfrom file:\n"<<readingfile<<endl;

    if(!reading){

      cout<<"can not open file"<<endl;

      exit(1);
    }

    int aux=-1;

    int naux=0;

    while (!reading.eof( )){

      reading>>aux;

      if(aux>-1){

      newblacklist.insert(aux);
    
      printf("%d \r",naux);

      ++naux;

      }
    }  
    cout<<endl;

  }

  string writingfile=folder;
  writingfile.append(argv[argc-2]);
  writingfile.append("_blacklist.txt");

  cout<<"writing new blacklist on file:\n"<<writingfile<<endl;

  ofstream writing(writingfile.c_str());

  if(!writing){

    cout<<"can not open file"<<endl;

    exit(1);
  }

  set<int>::const_iterator it;

  for ( it=newblacklist.begin() ; it != newblacklist.end(); it++ )

    writing<<(*it)<<"\n";



  return 0;
}
