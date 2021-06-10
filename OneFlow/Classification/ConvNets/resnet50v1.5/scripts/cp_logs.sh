NUM_NODES=$1
GPU_NUM_PER_NODE=$2
BSZ=$3
REPEAT_ID=$4

log_root=logs/oneflow
log_dir=$log_root/${NUM_NODES}n${GPU_NUM_PER_NODE}g

log_file=rn50_b${BSZ}_fp16_${REPEAT_ID}.log
summary_file=rn50_b${BSZ}_fp16_${REPEAT_ID}.csv

[ ! -d "${log_dir}" ] && mkdir -p ${log_dir}

cp ~/oneflow_temp/oneflow.log ${log_dir}/${log_file}
cp ~/oneflow_temp/log/summary.csv ${log_dir}/${summary_file}

# cp oneflow.INFO to log_dir 
#[ ! -d "${log_dir}/oneflow.INFO" ] && cp ~/oneflow_temp/log/VS002/oneflow.INFO ${log_dir}/oneflow.INFO
