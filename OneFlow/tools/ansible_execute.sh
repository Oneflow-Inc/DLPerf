password=$1
cmd=${2:-ls}

inventory_file=inventory

> $inventory_file
for ip in $(cat hosts); do
    if [[ $ip =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "$ip" ansible_ssh_pass=$password >> $inventory_file
    fi
done

# generate ansible.cfg
# echo '[defaults]' > ansible.cfg
# echo 'host_key_checking = False' >> ansible.cfg
# echo 'inventory = '$inventory_file >> ansible.cfg

# ansible all -m shell \
ansible all --inventory=$inventory_file -m shell \
    --ssh-extra-args "-o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \
    -a "$cmd"

# rm -f $inventory_file
