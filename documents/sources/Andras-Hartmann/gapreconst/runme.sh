mkdir data
cp script/* data/
cd data
./getdata.sh
./getmcdata.sh
./runc.slurm
