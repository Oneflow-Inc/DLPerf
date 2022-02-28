test_name=bsz_test
emb_size=16

for DEVICE_NUM_PER_NODE in 1 8
do
    for BATHSIZE in 16 64 256 1024 4096 16384 65536
    do
        bash dlrm_test.sh ${test_name} ${DEVICE_NUM_PER_NODE} ${BATHSIZE} ${emb_size}
    done
done
