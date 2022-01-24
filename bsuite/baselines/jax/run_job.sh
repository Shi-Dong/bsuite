#!/bin/bash
# run bsuite experiments

# base result directory
base_dir=/home/shidong/tmp/bsuite_results
# base directory of executables
base_exec=/home/shidong/bsuite-experiment/bsuite/baselines/jax

baseline=0
cpu=1
env_name=deep_sea_stochastic
bsuite_id=deep_sea_stochastic/4
ensemble=5
update_period=40
learning_rate=1e-4
penalty_weight=1e-7

if [ $baseline -eq 1 ]
then
    save_path=$base_dir/$env_name/ensemble_$ensemble/baseline
else
    save_path=$base_dir/$env_name/ensemble_$ensemble/update_period_$update_period/learning_rate_$learning_rate/penalty_weight_$penalty_weight
fi

for i in {1..1000}
do
    if [ ! -d "$save_path/run_$i" ]
    then
	save_path=$save_path/run_$i
	break
    fi
done

echo 'Result and log path: '
echo $save_path
mkdir -p $save_path

if [ $cpu -eq 1 ]
then
    use_cpu=True
else
    use_cpu=False
fi

if [ $baseline -eq 1 ]
then
    nohup python $base_exec/boot_dqn/run.py --num_ensemble=$ensemble --bsuite_id=$bsuite_id --verbose=True --overwrite=False --save_path=$save_path --cpu=$use_cpu &> $save_path/log &
else
    nohup python $base_exec/boot_dqn_new/run.py --num_ensemble=$ensemble --bsuite_id=$bsuite_id --verbose=True --overwrite=False --save_path=$save_path --learning_rate=$learning_rate --penalty_weight=$penalty_weight --target_update_period=$update_period --cpu=$use_cpu &> $save_path/log &
fi



