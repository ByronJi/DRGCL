#### GIN fine-tuning

nohup ./finetune_nocca_1e-4.sh bace 0 > log_bace_1e-4 2>&1 &
nohup ./finetune_nocca_1e-4.sh bbbp 0 > log_bbbp_1e-4 2>&1 &
nohup ./finetune_nocca_1e-4.sh clintox 1 > log_clintox_1e-4 2>&1 &
nohup ./finetune_nocca_1e-4.sh hiv 1 > log_hiv_1e-4 2>&1  &
nohup ./finetune_nocca_1e-4.sh muv 2 > log_muv_1e-4 2>&1 &
nohup ./finetune_nocca_1e-4.sh sider 2 > log_sider_1e-4 2>&1 &
nohup ./finetune_nocca_1e-4.sh tox21 3 > log_tox21_1e-4 2>&1 &
nohup ./finetune_nocca_1e-4.sh toxcast 3 > log_toxcast_1e-4 2>&1 &


