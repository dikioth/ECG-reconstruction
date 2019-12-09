#!/bin/bash

#do the all training for target in folder

#usage: trainforall.sh  signal1 signal2 ... signaln folder 


args=("$@")

numinputsignals=$(($#-1)) 

#numinputsignals-1=$(($numinputsignals-1))

aux=${args[@]:0:$numinputsignals}

aux2=${args[@]:1:$numinputsignals-1}

folder=${BASH_ARGV[0]}

for i in $aux
do

 echo "./train.sh "$i" "$folder" "

 ./train.sh "$i" "$folder"

done

echo "./train_aim $folder"

./train_aim.sh  $folder


gluedsignals=$1


if [ $numinputsignals -gt 1 ] ;

   then


   for i in $aux2 
     do


     gluedsignals="$gluedsignals$i"

   done

    echo "./glueseveralmlps ${aux[*]}  other$gluedsignals $folder"

    ./glueseveralmlps ${aux[*]}  other$gluedsignals $folder


    echo "./mergeblacklists ${aux[*]} other$gluedsignals $folder"

    ./mergeblacklists ${aux[*]} other$gluedsignals $folder

else

    echo "cp ../$folder/$1_blacklist.txt ../$folder/other$1_blacklist.txt"

    cp ../$folder/$1_blacklist.txt ../$folder/other$1_blacklist.txt


    echo "cp ../$folder/autoencoderbackpropweights_$1.txt ../$folder/autoencoderbackpropweights_other$1.txt"
    
    cp ../$folder/autoencoderbackpropweights_$1.txt ../$folder/autoencoderbackpropweights_other$1.txt
fi


echo "./trainotheraim_blacklist.sh other$gluedsignals aim $folder"

./trainotheraim_blacklist.sh other$gluedsignals aim $folder