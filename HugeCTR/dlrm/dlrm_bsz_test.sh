max_iter=12000
warmup_steps=1000
lr=0.5
for bsz in 16 64 256 1024 4096 16384 32768
do
    for ngpu in 1 8
    do 
        test_case=dlrm_test_n1g$ngpu}_bsz${bsz}
        mem_usage_file=${test_case}.mem

        python gpu_memory_usage.py 1> log/$mem_usage_file 2>&1 </dev/null &

        python dlrm.py \
            --gpu_num_per_node ${ngpu} \
            --eval_batchs 70 \
            --batch_size ${bsz} \
            --learning_rate ${lr} \
            --warmup_steps ${warmup_steps} \
            --max_iter ${max_iter} \
            --loss_print_every_n_iter 1000 \
            --embedding_vec_size 128 \
            --eval_interval 1000 | tee log/${test_case}.log
    done
done
