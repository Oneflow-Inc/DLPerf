
LOCAL_RUN=train_all_in_docker.sh
# ```
# ./$LOCAL_RUN $num_nodes $gpu_num_per_device $bsz_per_device $vocab_size $hidden_units_num $deep_vec_size $prefix $suffix
# ```


#for repeat_id in 1 2 3 4 5
for repeat_id in 2 3
do
    # case 2: vocab x 2
    # 1 node 1 device tests
    for vocab in 3200000 6400000 12800000 25600000 51200000 
    do
        ./$LOCAL_RUN 1 1 16384 $vocab 7 16 vocabX2 $repeat_id
    done

    # 1 node 8 devices tests
    for vocab in 3200000 6400000 12800000 25600000 51200000 102400000 204800000 409600000
    do
        ./$LOCAL_RUN 1 8 16384 $vocab 7 16 vocabX2 $repeat_id
    done

    # 4 nodes tests
    for vocab in 3200000 6400000 12800000 25600000 51200000 102400000 204800000 409600000 819200000 
    do
        ./$LOCAL_RUN 4 8 16384 $vocab 7 32 vocabX2 $repeat_id
    done
done

