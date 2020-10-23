
LOCAL_RUN=train_all_in_docker.sh
# ```
# ./$LOCAL_RUN $num_nodes $gpu_num_per_device $bsz $vocab_size $hidden_units_num $deep_vec_size $prefix $suffix
# ```


#for repeat_id in 1 2 3 4 5
for repeat_id in 1 2 3
do
    # case 1: bsz x 2
    # 1 node 1 device tests
    #for bsz in 512 1024 2048 4096 4096 8192 16384 32768 65536 131072 262144 524288
    for bsz in 512 1024 2048 4096 4096 8192 16384 32768 65536 131072 262144 
    do
        ./$LOCAL_RUN 1 1 $bsz 2322444 2 16 bszX2 $repeat_id
    done

    # 1 node 8 devices tests
    for bsz in 16384 32768 65536 131072 262144 524288 1048576 2097152
    do
        ./$LOCAL_RUN 1 8 $bsz 2322444 2 16 bszX2 $repeat_id
    done

    # 4 nodes tests
    #for bsz in 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608
    #for bsz in 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 
    #do
    #    ./$LOCAL_RUN 4 8 $bsz 2322444 2 32 bszX2 $repeat_id
    #done
done
