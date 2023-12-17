#### GIN fine-tuning
split=species

### for GIN
for runseed in 0 1 2 3 4 5 6 7 8 9
do
nohup python -u finetune_drgcl.py --model_file models_drgcl/drgcl_100.pth --split $split --epochs 50 --device 0 --runseed $runseed --gnn_type gin --lr 1e-4 > fine.log 2>&1 &
done
