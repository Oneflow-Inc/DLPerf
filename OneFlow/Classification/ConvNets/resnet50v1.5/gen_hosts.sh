set -x
if [[ $# -eq 0 ]]; then
    echo "example of usage: sh gen_bash.sh 32511 32473"
    echo "Illegal number of parameters" >&2
    exit 2
fi

ports=$@
dubhe_ip=10.105.0.1
#ports=(32511 32473)

# generate inventory file from port list
#host1 ansible_ssh_port=32511 ansible_ssh_host=10.105.0.1 ansible_ssh_user=root
#host2 ansible_ssh_port=32473 ansible_ssh_host=10.105.0.1 ansible_ssh_user=root
echo [dubhe] > inventory
for port in ${ports[@]}; do
  echo host_${port} ansible_ssh_port=${port} ansible_ssh_host=${dubhe_ip} ansible_ssh_user=root >> inventory
done

# generate ansible.cfg
echo '[defaults]' > ansible.cfg
echo 'log_path = hosts.stdout' >> ansible.cfg
echo 'host_key_checking = False' >> ansible.cfg
echo 'inventory = inventory' >> ansible.cfg

# remove old hosts.stdout if exists
rm -rf hosts.stdout

# collect ips by ansible
ansible dubhe \
  --ask-pass \
  -m shell \
  -a "hostname -I"

# generate hosts file
grep -Eo '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}' hosts.stdout > hosts

set +x

