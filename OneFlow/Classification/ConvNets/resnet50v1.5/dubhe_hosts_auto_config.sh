set -x

# install packages locally
apt update
apt install software-properties-common
add-apt-repository --yes --update ppa:ansible/ansible
apt install -y nmap ansible sshpass git vim

# scan nodes and generate inventory file for ansible
inventory_file=hosts
nmap -sP $@ | grep report | grep -Po '(?<=\().*(?=\))' > $inventory_file

# generate ssh key
ssh_dir=ssh
mkdir -p $ssh_dir
ssh-keygen -t rsa -N "" -f $ssh_dir/id_rsa
cat $ssh_dir/id_rsa.pub >> $ssh_dir/authorized_keys
chmod 600 $ssh_dir/authorized_keys

# ansible.cfg
echo '[defaults]' > ansible.cfg
echo 'host_key_checking = False' >> ansible.cfg

# copy ssh key to all hosts
ansible -i $inventory_file all \
  --ask-pass \
  -m copy \
  -a "src=$ssh_dir/ dest=/root/.ssh/ mode=0600 force=yes"

# remove usless ssh dir
rm -rf $ssh_dir

set +x

