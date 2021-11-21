cmd=${cmd:-ls}
num_nodes=${num_nodes:-0}
changedir=${changedir:-$PWD}

while [ $# -gt 0 ]; do
  case "$1" in
    --cmd=*)
      cmd="${1#*=}"
      ;;
    --num-nodes=*)
      num_nodes="${1#*=}"
      ;;
    --password=*)
      password="${1#*=}"
      ;;
    --chdir=*)
      changedir="${1#*=}"
      ;;
    *)
      printf "***************************\n"
      printf "* Error: Invalid argument.*\n"
      printf "***************************\n"
      exit 1
  esac
  shift
done

inventory_file=inventory

echo Create inventory file
# remove old inventory file and create an empty file
> $inventory_file

# append host to inventory file
i=0
while IFS= read -r ip
do
    if [[ $ip =~ ^#.* ]]; then
        continue
    fi
    if [[ $ip =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        if [ -z "$password" ]; then
            echo "$ip" >> $inventory_file
        else
            echo "$ip" ansible_ssh_pass=$password >> $inventory_file
        fi
        i=$((i+1))
        if [ $i -eq $num_nodes ]; then
            break
        fi
    fi
done  < hosts

ansible all --inventory=$inventory_file -m shell \
    -a "chdir=$changedir $cmd"
    # --ssh-extra-args "-o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \
    