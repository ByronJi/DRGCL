#### GIN fine-tuning

nohup ./finetune_nocca.sh bace 0 > log_bace 2>&1 &
nohup ./finetune_nocca.sh bbbp 0 > log_bbbp 2>&1 &
nohup ./finetune_nocca.sh clintox 1 > log_clintox 2>&1 &
nohup ./finetune_nocca.sh hiv 1 > log_hiv 2>&1  &
nohup ./finetune_nocca.sh muv 2 > log_muv 2>&1 &
nohup ./finetune_nocca.sh sider 2 > log_sider 2>&1 &
nohup ./finetune_nocca.sh tox21 3 > log_tox21 2>&1 &
nohup ./finetune_nocca.sh toxcast 3 > log_toxcast 2>&1 &


