#### GIN fine-tuning

nohup ./finetune.sh bace 0 > log_bace 2>&1 &
nohup ./finetune.sh bbbp 0 > log_bbbp 2>&1 &
nohup ./finetune.sh clintox 1 > log_clintox 2>&1 &
nohup ./finetune.sh hiv 1 > log_hiv 2>&1  &
nohup ./finetune.sh muv 2 > log_muv 2>&1 &
nohup ./finetune.sh sider 2 > log_sider 2>&1 &
nohup ./finetune.sh tox21 3 > log_tox21 2>&1 &
nohup ./finetune.sh toxcast 3 > log_toxcast 2>&1 &


