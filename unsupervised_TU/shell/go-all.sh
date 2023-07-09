#!/bin/bash -ex
cd ..
for dataset in MUTAG PROTEINS IMDB-BINARY DD COLLAB REDDIT-BINARY REDDIT-MULTI-5K NCI1
do
for seed in 0 1 2 3 4
do
 CUDA_VISIBLE_DEVICES=$1 python -u meta-mask-adam.py --DS $dataset --num-gc-layers 3 --aug $2 --seed $seed --max-epochs=$3 --weight-cl=$4  --weight-bl=$5 --criterion-type=$6 \
 # --disable-meta --disable-sigmoid --no-second-order
done
done
