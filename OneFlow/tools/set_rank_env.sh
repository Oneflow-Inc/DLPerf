export ONEFLOW_NODE_RANK=-1
local_ips=`hostname -I`
rank=0
while IFS= read -r ip
do
    if [[ $ip =~ ^#.* ]]; then
        continue
    fi
    for local_ip in ${local_ips}; do
        # echo $ip, $local_ip
        if [[ $ip == $local_ip ]]; then
            # echo got it
            export ONEFLOW_NODE_RANK=$rank
            break
        fi
    done
    rank=$((rank+1))
done < hosts
