#### GIN fine-tuning
split=scaffold
dataset=$1

for runseed in 0 1 2 3 4 5 6 7 8 9
do
model_file=${unsup}
python finetune.py --model_file models_drgcl/drgcl_100.pth --split $split --runseed $runseed --gnn_type gin --dataset $dataset --lr 1e-3 --epochs 100 --device $2
done