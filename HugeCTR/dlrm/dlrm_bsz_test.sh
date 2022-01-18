max_iter=12000
warmup_steps=1000
lr=0.5
for bsz in 16 64 256 1024 4096 16384 65536
do
    test_case=dlrm_test_bsz${bsz}

    python dlrm.py \
        --gpu_num_per_node 8 \
        --data_dir /dataset/criteo_kaggle/hugectr_dlrm \
        --eval_batchs 70 \
        --batch_size ${bsz} \
        --learning_rate ${lr} \
        --warmup_steps ${warmup_steps} \
        --max_iter ${max_iter} \
        --loss_print_every_n_iter 100 \
        --embedding_vec_size 128 \
        --eval_interval 100 | tee log/${test_case}.log
done
