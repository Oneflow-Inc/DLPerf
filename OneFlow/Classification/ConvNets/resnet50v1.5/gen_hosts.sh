set -x
if [[ $# -le 1 ]]; then
    echo "example of usage: sh gen_bash.sh passowrd 32511 32473"
    echo "Illegal number of parameters" >&2
    exit 2
fi

password=$1
ports=${@:2}

dubhe_ip=10.105.0.1
#ports=(32511 32473)

# generate inventory file from port list
#host1 ansible_ssh_port=32511 ansible_ssh_host=10.105.0.1 ansible_ssh_user=root
#host2 ansible_ssh_port=32473 ansible_ssh_host=10.105.0.1 ansible_ssh_user=root
echo [dubhe] > inventory
for port in ${ports[@]}; do
  echo host_${port} ansible_ssh_port=${port} ansible_ssh_host=${dubhe_ip} ansible_ssh_user=root ansible_ssh_pass=$password >> inventory
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
  -m shell \
  -a "hostname -I"

  #--ask-pass \
# generate hosts file
grep -Eo '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}' hosts.stdout > hosts

set +x

