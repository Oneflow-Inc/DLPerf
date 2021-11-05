cmd=${cmd:-ls}

while [ $# -gt 0 ]; do
  case "$1" in
    --cmd=*)
      cmd="${1#*=}"
      ;;
    --password=*)
      password="${1#*=}"
      ;;
    *)
      printf "***************************\n"
      printf "* Error: Invalid argument.*\n"
      printf "***************************\n"
      exit 1
  esac
  shift
done

if [ -z "$password" ]; then
    inventory_file=hosts
else
    echo Create inventory file with password
    inventory_file=inventory
    # remove old inventory file and create an empty file
    > $inventory_file

    # append host to inventory file
    for ip in $(cat hosts); do
        if [[ $ip =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            echo "$ip" ansible_ssh_pass=$password >> $inventory_file
        fi
    done
fi

ansible all --inventory=$inventory_file -m shell \
    --ssh-extra-args "-o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \
    -a "$cmd"
