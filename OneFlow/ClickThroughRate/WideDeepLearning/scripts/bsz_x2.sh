LOCAL_RUN=$1
prefix=$2
suffix=$3
# ```
# ./$LOCAL_RUN $num_nodes $gpu_num_per_device $bsz $vocab_size $hidden_units_num $deep_vec_size $prefix $suffix
# ```

# 1 node 1 device tests
for bsz in 512 1024 2048 4096 4096 8192 16384 32768 65536 131072 262144 524288
do
    $LOCAL_RUN 1 1 $bsz 2322444 2 16 $prefix $suffix
done

# 1 node 8 devices tests
for bsz in 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152
do
    $LOCAL_RUN 1 8 $bsz 2322444 2 16 $prefix $suffix
done
