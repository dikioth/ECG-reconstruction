#makefile to compile all c++ programs used here


#change it acordingly with your GSL instalation
GSLPATH=/home/rui/neuralnets/c++/gsl

#change g++ if you use another compiler and substitute the compiler switch -fopenmp by the appropriate one  
CC = g++ 

CFLAGS = -O2 -fopenmp  -I$(GSLPATH)  -Wall

LFLAGS = -fopenmp -L$(GSLPATH)/lib 


all: backpropautoencodergsl backpropmlp1layer backpropmlp4layers create_autoencoder_blacklist fwdhiddenlayerrbmvislinear fwdmlp4layers fwdmlp_data_to1stlayer_gsl fwdmlp_data_to2ndlayer_gsl glueseveralmlps mergeblacklists prepareautoencoder trainboltzmannperceptron traincd1rbmlogistic traincd1rbmvislinear trainmeanfieldperceptron trainmeanfieldrbmlogistic trainmeanfieldrbmvislinear 

TARGETS = backpropautoencodergsl backpropmlp1layer backpropmlp4layers create_autoencoder_blacklist fwdhiddenlayerrbmvislinear fwdmlp4layers fwdmlp_data_to1stlayer_gsl fwdmlp_data_to2ndlayer_gsl glueseveralmlps mergeblacklists prepareautoencoder trainboltzmannperceptron traincd1rbmlogistic traincd1rbmvislinear trainmeanfieldperceptron trainmeanfieldrbmlogistic trainmeanfieldrbmvislinear

OBJS = backpropautoencodergsl.o blacklist.o parallelgradgsl.o netdimsandfilenames.o iogslvectormatrix.o checkfstream.o backpropmlp1layer.o backpropmlp4layers.o create_autoencoder_blacklist.o fwdhiddenlayerrbmvislinear.o netrbm.o fwdmlp4layers.o fwdmlp_data_to1stlayer_gsl.o fwdmlp_data_to2ndlayer_gsl.o glueseveralmlps.o mlp.o glue2mlps.o mergeblacklists.o prepareautoencoder.o trainboltzmannperceptron.o traincd1rbmlogistic.o traincd1rbmvislinear.o trainmeanfieldperceptron.o trainmeanfieldrbmlogistic.o trainmeanfieldrbmvislinear.o reconstructfrompatchesstartingfrom.o interpolate.o


.cpp.o: $<
	$(CC)   -c $(CFLAGS) $< -o $@

backpropautoencodergsl : backpropautoencodergsl.o blacklist.o parallelgradgsl.o netdimsandfilenames.o iogslvectormatrix.o checkfstream.o
	$(CC)  $(LFLAGS) $^ -o $@ -lgsl -lgslcblas -lm

backpropmlp1layer : backpropmlp1layer.o parallelgradgsl.o netdimsandfilenames.o iogslvectormatrix.o blacklist.o checkfstream.o
	$(CC)  $(LFLAGS) $^ -o $@ -lgsl -lgslcblas -lm


backpropmlp4layers : backpropmlp4layers.o blacklist.o parallelgradgsl.o netdimsandfilenames.o iogslvectormatrix.o checkfstream.o
	$(CC)  $(LFLAGS) $^ -o $@ -lgsl -lgslcblas -lm

create_autoencoder_blacklist : create_autoencoder_blacklist.o blacklist.o parallelgradgsl.o netdimsandfilenames.o iogslvectormatrix.o checkfstream.o
	$(CC)  $(LFLAGS) $^ -o $@ -lgsl -lgslcblas -lm

fwdhiddenlayerrbmvislinear : fwdhiddenlayerrbmvislinear.o netrbm.o netdimsandfilenames.o iogslvectormatrix.o checkfstream.o
	$(CC)  $(LFLAGS) $^ -o $@ -lgsl -lgslcblas -lm

fwdmlp4layers : fwdmlp4layers.o reconstructfrompatchesstartingfrom.o interpolate.o parallelgradgsl.o netdimsandfilenames.o iogslvectormatrix.o checkfstream.o
	$(CC)  $(LFLAGS) $^ -o $@ -lgsl -lgslcblas -lm

fwdmlp_data_to1stlayer_gsl : fwdmlp_data_to1stlayer_gsl.o parallelgradgsl.o iogslvectormatrix.o netdimsandfilenames.o checkfstream.o
	$(CC)  $(LFLAGS) $^ -o $@ -lgsl -lgslcblas -lm

fwdmlp_data_to2ndlayer_gsl : fwdmlp_data_to2ndlayer_gsl.o parallelgradgsl.o iogslvectormatrix.o netdimsandfilenames.o checkfstream.o 
	$(CC)  $(LFLAGS) $^ -o $@ -lgsl -lgslcblas -lm

glueseveralmlps : glueseveralmlps.o mlp.o glue2mlps.o iogslvectormatrix.o netdimsandfilenames.o checkfstream.o
	$(CC)  $(LFLAGS) $^ -o $@ -lgsl -lgslcblas -lm

mergeblacklists : mergeblacklists.o
	$(CC)  $(LFLAGS) $^ -o $@ -lgsl -lgslcblas -lm

prepareautoencoder : prepareautoencoder.o netdimsandfilenames.o netdimsandfilenames.o iogslvectormatrix.o checkfstream.o
	$(CC)  $(LFLAGS) $^ -o $@ -lgsl -lgslcblas -lm

trainboltzmannperceptron : trainboltzmannperceptron.o parallelgradgsl.o netdimsandfilenames.o iogslvectormatrix.o blacklist.o checkfstream.o
	$(CC)  $(LFLAGS) $^ -o $@ -lgsl -lgslcblas -lm

traincd1rbmlogistic : traincd1rbmlogistic.o netrbm.o blacklist.o netdimsandfilenames.o iogslvectormatrix.o checkfstream.o
	$(CC)  $(LFLAGS) $^ -o $@ -lgsl -lgslcblas -lm

traincd1rbmvislinear : traincd1rbmvislinear.o blacklist.o netdimsandfilenames.o iogslvectormatrix.o checkfstream.o
	$(CC)  $(LFLAGS) $^ -o $@ -lgsl -lgslcblas -lm

trainmeanfieldperceptron : trainmeanfieldperceptron.o blacklist.o parallelgradgsl.o netdimsandfilenames.o iogslvectormatrix.o checkfstream.o
	$(CC)  $(LFLAGS) $^ -o $@ -lgsl -lgslcblas -lm

trainmeanfieldrbmlogistic : trainmeanfieldrbmlogistic.o netrbm.o blacklist.o netdimsandfilenames.o iogslvectormatrix.o checkfstream.o
	$(CC)  $(LFLAGS) $^ -o $@ -lgsl -lgslcblas -lm

trainmeanfieldrbmvislinear : trainmeanfieldrbmvislinear.o blacklist.o parallelgradgsl.o netdimsandfilenames.o iogslvectormatrix.o checkfstream.o
	$(CC)  $(LFLAGS) $^ -o $@ -lgsl -lgslcblas -lm


clean :
	rm -f $(TARGETS) $(OBJS)
