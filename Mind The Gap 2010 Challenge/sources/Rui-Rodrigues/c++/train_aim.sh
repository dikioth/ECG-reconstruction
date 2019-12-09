#!/bin/bash

#usage train_aim.sh folder

echo " runing  traincd1rbmvislinear start aim "$1" "

./traincd1rbmvislinear start aim "$1"
 
echo " runing  trainmeanfieldrbmvislinear aim "$1" "

./trainmeanfieldrbmvislinear aim "$1"

echo " runing  fwdhiddenlayerrbmvislinear aim "$1" "

./fwdhiddenlayerrbmvislinear aim $1 


echo " runing  traincd1rbmlogistic start aim "$1" "

./traincd1rbmlogistic start aim $1

echo " runing  trainmeanfieldrbmlogistic aim "$1" "

./trainmeanfieldrbmlogistic aim $1 

echo " runing  prepareautoencoder aim "$1" "

./prepareautoencoder aim $1

echo " runing  backpropautoencodergsl cont $1"

./backpropautoencodergsl cont aim $1 


