
bash 300k_iters.sh $repeat_id
bash 500_iters.sh $repeat_id
for repeat_id in 1 2 3 4 5
do
    bash fix_bsz_per_device_1_node_run.sh $repeat_id
    bash fix_bsz_per_device_mul_nodes_run.sh $repeat_id

    bash fix_total_bsz_1_node_run.sh $repeat_id
    bash fix_total_bsz_mul_nodes_run.sh $repeat_id

    bash vocab_x2_1_device_run.sh $repeat_id
    bash vocab_x2_1_node_run.sh $repeat_id
    bash vocab_x2_mul_nodes_run.sh $repeat_id

    bash bsz_x2_1_device_run.sh $repeat_id
    bash bsz_x2_1_node_run.sh $repeat_id
    bash bsz_x2_mul_nodes_run.sh $repeat_id
done


