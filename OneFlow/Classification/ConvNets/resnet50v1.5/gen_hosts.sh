nmap -sP $@ | grep report | grep -Po '(?<=\().*(?=\))' > hosts

