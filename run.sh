#!/bin/bash

# default values
nc=10 #number of clients
nro=100 #number of rounds
nru=4 #number of runs
so=0 #seed offset
strategy=0 #flower server strategy (aggregation setup)
dset="MNIST_2class" #dataset to be used
lr=0.5 #learning rate
ci=False #class imbalance
si=False #sample imbalance
mc="FNN" #model choice
sf=True #save file
fol="../afstuderen/datafiles/" #save folder
lepo=1 #local epochs
lbat=1 #local batches


#opt_str=nc,nro,nru,so,str,dset,lr,ci,si,mc,sf,fol,lepo,lbat
#parse input
#in=$(getopt --long ${opt_str}: -- "$@")
#in=$(getopt -o --long nc: -- "$@")

#eval set -- "$in"



while true;
do
    case "$1" in
        --nc)  nc=$2; shift 2 ;;
        --nro) nro="$2"; shift 2 ;;
        --nru) nru="$2"; shift 2 ;;
        --so) so="$2"; shift 2 ;;
        --str) strategy="$2"; shift 2 ;;
        --dset) dset="$2"; shift 2 ;;
        --lr) lr="$2"; shift 2 ;;
        --ci) ci="$2"; shift 2 ;;
        --si) si="$2"; shift 2 ;;
        --mc) mc="$2"; shift 2 ;;
        --sf) sf="$2"; shift 2 ;;
        --fol) fol="$2"; shift 2 ;;
        --lepo) lepo="$2"; shift 2 ;;
        --lbat) lbat="$2"; shift 2 ;;
        -- ) shift; break ;;
        * ) break ;;
    esac
done

#echo "${nc}"

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate flower


python run_flower_server.py --nro=${nro} --strat=${strategy} --nc=${nc} &


for ((i = 0; i<=$nc-1; i++))
do
    sleep 5
    python run_flower_client.py --cid $i --dset $dset --ci $ci --si $si --lr $lr --mc $mc --lepo $lepo --lbat $lbat &
done

exit