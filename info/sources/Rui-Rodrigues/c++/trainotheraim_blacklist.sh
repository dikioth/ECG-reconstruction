#!/bin/bash
#calling backpropmlp4layers

#usage trainotheraim_blacklist.sh othername aim folder

echo " runing  fwdmlp_data_to2ndlayer_gsl $1 $3"

./fwdmlp_data_to2ndlayer_gsl $1 $3

echo " runing  fwdmlp_data_to2ndlayer_gsl $2 $3"

./fwdmlp_data_to2ndlayer_gsl $2 $3


echo " runing  trainboltzmannperceptron start $1 $2 $3 useblacklist"

./trainboltzmannperceptron start $1 $2 $3 useblacklist

echo " runing  trainmeanfieldperceptron $1 $2 $3 useblacklist"

./trainmeanfieldperceptron $1 $2 $3 useblacklist


echo " runing  fwdmlp_data_to1stlayer_gsl $1 $3"

./fwdmlp_data_to1stlayer_gsl $1 $3


echo " runing  backpropmlp1layer startperceptron $1 $2 $3 useblacklist"

./backpropmlp1layer startperceptron $1 $2 $3 useblacklist


echo " runing  backpropmlp4layers startmlp1 $1 $2 $3 useblacklist"

./backpropmlp4layers startmlp1 $1 $2 $3 useblacklist
