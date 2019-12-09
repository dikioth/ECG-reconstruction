
     /* netrbm.h - class restricted boltzmann machine (rbm)
        Copyright (C) 2010  Rui Rodrigues <rapr@fct.unl.pt>
        This software is released under the terms of the GNU General
        Public License (http://www.gnu.org/copyleft/gpl.html).
     */


#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>



class netrbm{

public:

  netrbm(unsigned ninputs,unsigned nhidden, unsigned numcases,
	 unsigned Maxnumthreads,unsigned Blocksize, unsigned Npatches,
	 gsl_vector *Vectorweightsandbias);

  ~netrbm();


  //data

  gsl_matrix  *permutedtdata, *hiddenlayer, * batchimagedata, *posprobs, *negprobs, *hiddenactivation,
    *posprods, *negprods, *deltapureweights;

  gsl_vector *vectorweightsandbias,*deltavisbias, *deltahidbias, *sumbatchtdata, *sumbatchimagedata, *sumposprobs, *sumnegprobs;

  gsl_matrix_view  batchtdata, *blocktdata, *blockposprobs,  *blocknegprobs, *blockimagedata, *blockhiddenactivation,pureweights;

  gsl_vector_view visbias, hidbias;

  unsigned ninputs, nhidden, numcases, maxnumthreads, *sizesblocks, blocksize, npatches;

};
