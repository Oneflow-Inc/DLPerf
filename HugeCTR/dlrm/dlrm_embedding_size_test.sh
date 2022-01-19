max_iter=12000
warmup_steps=1000
lr=0.5
for embedding_vec_size in 2 8 32 128
do
        for ngpu in 1 8
        do
                test_case=dlrm_test_n1g${ngpu}_embsz${embedding_vec_size}
                mem_usage_file=${test_case}.mem

                python gpu_memory_usage.py 1> log/$mem_usage_file 2>&1 </dev/null &

                python dlrm.py \
                        --gpu_num_per_node ${ngpu} \
                        --eval_batchs 70 \
                        --batch_size 32768 \
                        --learning_rate ${lr} \
                        --warmup_steps ${warmup_steps} \
                        --max_iter ${max_iter} \
                        --loss_print_every_n_iter 100 \
                        --embedding_vec_size ${embedding_vec_size} \
                        --eval_interval 100 | tee log/${test_case}.log
        done
done

