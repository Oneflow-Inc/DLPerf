cmd=${1:-ls}

ansible all --inventory=inventory -m shell \
    --ask-pass \
    --ssh-extra-args "-o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \
    -a "$cmd"
