test_name=embsize_test
BATHSIZE =32

for DEVICE_NUM_PER_NODE in 1 8
do
    for emb_size in 2 8 32
    do
        bash dlrm_test.sh ${test_name} ${DEVICE_NUM_PER_NODE} ${BATHSIZE} ${emb_size}
    done
done
