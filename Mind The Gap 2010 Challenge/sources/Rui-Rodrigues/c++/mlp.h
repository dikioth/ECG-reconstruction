
     /* mlp.h - class defining a multilayer mlp and methods to read and write it on a file
        Copyright (C) 2010  Rui Rodrigues <rapr@fct.unl.pt>
        This software is released under the terms of the GNU General
        Public License (http://www.gnu.org/copyleft/gpl.html).
     */



#include <gsl/gsl_matrix.h>
#include<vector>


using namespace std;


struct mlp{

  unsigned numhidlayers;

  vector<unsigned> dimlayers;

  vector<unsigned> dimmatrices;

  vector<gsl_matrix*> mlpmatrix;

  


  //constructors

  mlp(unsigned numhidlayers, vector<unsigned>& dimlayers);

  mlp(unsigned numhidlayers, vector<unsigned>& dimlayers,vector<gsl_matrix*>& mlpmatrix);

  mlp(const mlp& mlpother);

  ~mlp();

  void operator=(const mlp&);

  void getweightsfromgslvector(gsl_vector* container);
  
  void loadtxtweights(const char* filename);

  void savetxtweights( const char* filename);

  void init(const unsigned Numhidlayers,const vector<unsigned>& Dimlayers,const vector<gsl_matrix*>& Mlpmatrix);

};
