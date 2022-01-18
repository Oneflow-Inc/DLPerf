max_iter=12000
warmup_steps=1000
lr=0.5
for embedding_vec_size in 2 8 32 128 512
do
    test_case=dlrm_test_embsz${embedding_vec_size}
    mem_usage_file=${test_case}.mem

    python gpu_memory_usage.py 1> log/$mem_usage_file 2>&1 </dev/null &

    python dlrm.py \
        --gpu_num_per_node 8 \
        --data_dir /dataset/criteo_kaggle/hugectr_dlrm \
        --eval_batchs 70 \
        --batch_size 65536 \
        --learning_rate ${lr} \
        --warmup_steps ${warmup_steps} \
        --max_iter ${max_iter} \
        --loss_print_every_n_iter 1000 \
        --embedding_vec_size ${embedding_vec_size} \
        --eval_interval 1000 | tee log/${test_case}.log
done
