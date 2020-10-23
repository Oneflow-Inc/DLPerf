
LOCAL_RUN=train_all_in_docker.sh
# ```
# ./$LOCAL_RUN $num_nodes $gpu_num_per_device $bsz_per_device $vocab_size $hidden_units_num $deep_vec_size $prefix $suffix
# ```
declare -a num_nodes_list=(4 2 1 1 1 1)
declare -a num_gpus_list=(8 8 8 4 2 1)
len=${#num_nodes_list[@]}


#for repeat_id in 1 2 3 4 5
for repeat_id in 1 2 3
do
    # case 2: vocab x 2
    # 1 node 1 device tests
    # xuan for vocab in 3200000 6400000 12800000 25600000 51200000
    # xuan do
    # xuan     ./$LOCAL_RUN 1 1 16384 $vocab 7 16 vocabX2 $repeat_id
    # xuan done

    # 1 node 8 devices tests
    #for vocab in 25600000 51200000 102400000 204800000 409600000
    for vocab in 3200000 6400000 12800000
    do
        ./$LOCAL_RUN 1 8 16384 $vocab 7 16 vocabX2 $repeat_id
    done

    # 4 nodes tests
    #for vocab in 204800000 409600000 819200000 1638400000
    for vocab in 3200000 6400000 12800000 25600000 51200000 102400000
    do
        ./$LOCAL_RUN 4 8 16384 $vocab 7 32 vocabX2 $repeat_id
    done
done

