#!/bin/bash
#calling backpropmlp4layers

#usage train.sh signal folder

echo " runing  traincd1rbmvislinear start "$1" "$2" "

./traincd1rbmvislinear start "$1" "$2"

#for ((  i = 0 ;  i <= $4;  i++  ))
#do
 
echo " runing  trainmeanfieldrbmvislinear "$1" "$2" "

./trainmeanfieldrbmvislinear "$1" "$2"

echo " runing  fwdhiddenlayerrbmvislinear $1 $2"

./fwdhiddenlayerrbmvislinear $1 $2


echo " runing  traincd1rbmlogistic $1 $2"

./traincd1rbmlogistic start $1 $2

echo " runing  trainmeanfieldrbmlogistic $1 $2"

./trainmeanfieldrbmlogistic $1 $2

echo " runing  prepareautoencoder $1 $2"

./prepareautoencoder $1 $2

echo " runing  backpropautoencodergsl cont $1 $2"

./backpropautoencodergsl cont $1 $2


echo " runing create_autoencoder_blacklist $1 $2"

./create_autoencoder_blacklist $1 $2

#done

