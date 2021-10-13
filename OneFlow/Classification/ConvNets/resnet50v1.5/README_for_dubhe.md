### gen_hosts.sh
该脚本运行在非容器环境，比如用户自己的机器上，需要安装ansible和sshpass。
通常天枢专业版提供了ssh命令帮助用户远程连接到已经申请的一组节点，本脚本帮助获取这组节点的内网IP。
脚本的运行依赖输入密码和一组端口号，例如:
```
sh gen_bash.sh password 32511 32473
```
脚本将根据这些端口号自动生成一个`inventory`文件，并且生成一个`ansible.cfg`文件供后面的ansible命令使用。ansible命令会在每个节点上运行"hostname -I"命令得到节点内网的ip地址，返回本地后重新处理一下保存到`hosts`文件中。

### gen_hosts_on_host.sh
该脚本应该运行在容器内部。
脚本使用`nmap`自动检测同组的机器，并保存至名为`hosts`的文件。
脚本的运行依赖输入，例如：
```
bash dubhe_hosts_auto_config.sh 10.244.1.* 10.244.237.*
```
目前还没有方法自动获得所有待搜索的网段，所以必须依赖用户输入。

`hosts`文件将被用于后面的测试，帮助生成参与测试节点的ip列表。

### dubhe_hosts_auto_config.sh
新建连接后进入master节点，运行此脚本，脚本将安装并使用`nmap`自动检测同组的机器，并保存至名为`hosts`的文件。
脚本的运行依赖输入，例如：
```
bash dubhe_hosts_auto_config.sh 10.244.1.* 10.244.237.*
```
目前还没有方法自动获得所有待搜索的网段，所以必须依赖用户输入。

该脚本还做了如下工作:
- 在主机安装ansible, sshpass, git, vim等软件包
- 生成一组 ssh key，并写入authorized_keys
- 然后把生成的`ssh`目录通过ansible拷贝到`hosts`文件里有的主机的`.ssh`目录，从而实现各个机器之间免密。

### ansible_launch_on_host.sh
该脚本用于自动化测试。本脚本依赖`gen_hosts.sh`或`gen_hosts_on_host.sh`生成的`hosts`文件
结合测试的需求，脚本的主体是4重循环，每一重循环都代表了测试的一周需求，从外到内分别是：

1. `for amp in 0 1`代表了是否打开混合精度进行测试
2. `for bsz in ${bsz_list[@]}`因为混合精度的开关会影响显存使用，进而影响最大batch size，所以设置了不同的batch size列表进行测试
3. `for (( i=0; i<$len; i++ ))`我们会在不同的资源条件下进行测试，比如单机单卡、4机8卡（共计32卡）等等，所以定义了两个list用来表示希望测试的规模，`num_nodes_list`代表了将采用多少台服务器进行测试，`num_gpus_list`表示每台服务器有几块GPU卡。这两个list的长度必须一样，可以通过同时修改这两个list来确定测试的规模
4. `for (( j=0; j<$REPEAT_TIMES; j++ ))`每一个测试都可以重复多次，`REPEAT_TIMES`定义了重复的次数。

脚本的核心是命令`cmd`的生成，也就是给`ansible`命令设置参数然后运行，说明如下：

- `ansible all -i ${hosts},` ${hosts}保存的是从`hosts`文件获取的ip列表并截取了其中num_nodes个ip
- `-m shell`表示选择的是`shell` 模块，后面的`-a`是shell模块将使用的参数
- `chdir=${SHELL_DIR}`指定了shell命令运行的初始位置
- `"bash local_train.sh ${num_nodes} ${num_gpus} ${bsz} ${amp} ${j} ${hosts}"`我们单独定义了一个`local_train.sh`脚本，该脚本的运行需要传入节点数、gpu数、批次大小等参数，而这些参数就是前面的各重循环确定的。

