max_iter=12000
warmup_steps=1000
lr=0.5
test_case=dlrm_baseline_${max_iter}_${warmup_steps}_${lr}

python dlrm_kaggle_fp32.py \
       --gpu_num_per_node 8 \
       --eval_batchs 70 \
       --max_iter ${max_iter} \
       --batch_size 65536 \
       --learning_rate ${lr} \
       --warmup_steps ${warmup_steps} \
       --loss_print_every_n_iter 100 \
       --eval_interval 100 | tee log/${test_case}.log