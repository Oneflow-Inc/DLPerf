# OneFlow Benchmark test tools

## `ansible_execute.sh`
该脚本用于帮助在集群中执行一个命令，比如`ls`。脚本依赖一个名为`hosts`的文件，里面保存了集群的ip列表，比如：
```
# ip list
10.105.0.32
10.105.0.33
10.105.0.34
10.105.0.35
```

该脚本依赖三个参数：
- `--cmd`: 待执行的命令，不选就执行`ls`命令
- `--num-nodes`: 需要在几台节点上执行，不选表示在所有hosts都执行
- `--password`: ansible执行ssh访问其他节点时所需要键入的密码，不选需要配置免密登陆

脚本的工作原理：脚本首先获取输入的参数，然后根据`hosts`文件创建一个`inventory`文件供`ansible`命令使用，`inventory`中根据`--num-nodes`限定了节点数量，如果有配`--password`就会为每个节点设置`ansible_ssh_pass`，最后执行`--cmd`指定的命令。

例子🌰：
```bash
./ansible_execute.sh \
  --password=******** \
  --cmd="python3 -m pip install -U /path/to/oneflow.whl"
```
上面的例子中，需要注意的是`--cmd`的值被引号包裹，不然解析会出问题；另外`python3`通常会被绝对路径替换，比如`/home/tianshu/miniconda3/bin/python3`以避免各个机器的环境不一致。

## `run_resnet_tests.py`
该脚本用于在多个节点上执行一组resnet50的性能测试。同样依赖前面提到的`hosts`文件。

该脚本依赖下面的参数：
- `--python_bin`: python可执行文件的位置，比如`/home/tianshu/miniconda3/bin/python3`
- `--script`: resnet50脚本的位置，即[OneFlow-Benchmark](https://github.com/Oneflow-Inc/OneFlow-Benchmark)项目`Classification/cnns/of_cnn_train_val.py`文件的位置
- `--data_dir`: imagenet ofrecord格式数据集的位置
- `--log_dir`: 待输出log的保存目录
- `--repeat`: 这组测试重复执行的次数

例子🌰：
```bash
python3 run_resnet_tests.py \
  --password=******** \
  --python_bin=`which python3` \
  --script=/nfs/git-repos/OneFlow-Benchmark/Classification/cnns/of_cnn_train_val.py \
  --data_dir=/nfs/data/imagenet/ofrecord \
  --log_dir=log
```

### 脚本实现说明
#### `OfResnetTest`类
该类继承自`GroupTest`，详见`group_test.py`。对于一组测试来说，需要知道：
- 测试需要使用的`python_bin`的位置
- 待测试的脚本`script`
- 测试脚本所需的参数`args`
- 测试是需要设置的环境变量`envs`
- 测试需要用到的主机ip列表`hosts`
- 已经测试结果保存的目录`log_dir`

##### 测试策略矩阵 strategy.matrix
`GroupTest`的核心概念就测试策略矩阵`metrix`，想法来自[github workflow strategy](https://docs.github.com/en/actions/learn-github-actions/workflow-syntax-for-github-actions#jobsjob_idstrategy)。每个测试案例独特的配置值以字典的形式存储在矩阵（其实是一个`list`）中，测试时遍历该矩阵，逐个执行测试。`GroupTest`提供了`append_matrix`接口帮助生成矩阵。

##### `__call__`与`run_once`函数
`GroupTest`重载了`__call__`函数用来生成可以运行的命令列表`cmds`。其中根据`repeat`来决定要执行几次`run_once`。

`run_once`负责根据`matrix`生成一组测试命令。

##### 命名规则 `naming_rule`
`get_log_name`负责根据测试的一组参数生成特定的日志文件名。考虑到参数名称一般都比较长，而且大部分参数都一样，所以也是定义了一个字典`naming_rule`，key是参数名，value是缩写，只有字典里的参数参与命名，比如：
```python
naming_rule = {
    'num_nodes': 'n',
    'gpu_num_per_node': 'g',
    'batch_size_per_device': 'b',
    'use_fp16': 'amp',
}
```

`set_log_naming_rule`用来设置命名规则。

#### `init_tests`
前面介绍的`GroupTest`类，`init_tests`就是构造组测试类对象的函数，其中包括：
- `envs`
- `default_args`: 每个测试都包括的缺省参数
- `runs_on = [[1, 1], [1, 8], [2, 8], [4, 8]]`: 测试需要使用的资源，是一个列表，每个元素是一个两元列表（tuple也行），代表了节点数和每个节点的卡数。比如这里对应着1机1卡、1机8卡、2机8卡、4机8卡。
- `for run_on in runs_on`这个循环就是为了给不同的设备配置设置不同的参数。另外其中，是否开启混合精度选项（`use_fp16`）还需要配置不同的batch size。

其实这里可以是很灵活的，用户可以根据测试需求组织测试参数，扩展`matrix`。

#### 生成测试命令
`cmds = rn50(FLAGS.repeat)` 只是生成了一组测试命令，并不真正执行测试。`cmds`是一个list，每个元素由两个子元素组成：执行该命令的节点数`num_nodes`和命令`cmd`本身。

#### 执行测试
`exec_cmd`是真正的执行某一项测试。
执行的时候，需要根据节点数先生成一个临时的`inventory`文件。然后把测试命令保存成一个临时的shell脚本文件`tmp_run.sh`，主要是因为不保存临时文件总数报错，没找到原因。然后生成ansible命令，通过`os.system`的方式执行。

## `run_bert_tests.py`
与`run_resnet_tests.py`类似。
例子🌰：
```python
python3 run_bert_tests.py \
  --password=******** \
  --python_bin=`which python3` \
  --script=/nfs/git-repos/OneFlow-Benchmark/LanguageModeling/BERT/run_pretraining.py \
  --data_dir=/nfs/data/wiki_seq_len_128 \
  --log_dir=log
```

## 生成测试报告
`generate_test_results.py`负责生成测试报告，它依赖下面一些参数：
- `--log_dir`: 保存日志的目录地址
- `--endswith`: 日志的后缀，缺省是`.log`
- `--contains`: 用于过滤日志用的字符串，采用只包含此字符串的日志文件
- `--output`: 测试报告结果输出的文件名，目前是`;`分割的csv文件
- `--type`: 日志类型，目前支持`cnn`和`bert`两种类型，缺省是`cnn`
- `--start_iter`和`--end_iter`: 目前的日志中会打印时间戳，选取两个特定的iter时间戳来计算吞吐（throughput）和batch时延（latency）。
- `--master_ip`: 目前只提取主机的性能数据，这里需要指定一下主机的ip，该ip会被编入查询消息查询`--start_iter`和`--end_iter`时间段内该节点的性能指标。

例子🌰：
1. 提取resnet50的测试结果
```bash
python3 generate_test_results.py --contains=A100-node26
```
2. 提取bert的测试结果
```bash
python3 generate_test_results.py --type=bert --start_iter=39 --end_iter=139 --contains=A100-node26
```

## `exporter_util.py`
用来提取指定机器、指定时间段内的机器性能指标。可以通过修改`metric_info_dict`来决定提取哪些指标，目前包括：
```python
metric_info_dict = {
  'DCGM_FI_DEV_FB_USED': (9440, "prod-gpu", 'GPU memory used (in MiB).'),
  'DCGM_FI_DEV_POWER_USAGE': (9440, "prod-gpu", 'GPU power usage (in W).'),
  'DCGM_FI_PROF_GR_ENGINE_ACTIVE': (9440, "prod-gpu", 'GPU utilization (in %).'),
  'node_memory_Committed_AS_bytes': (9100, "prod-node", 'Node Memory information field Committed (in bytes)'),
}
```
这是一个字典，key是指标的名称，value是一个三元组，分别对应着：
- 查询的端口号，9440对应的是GPU相关的端口号，9100是cpu主机相关的端口号
- 过滤用的`job`名字，因为查询返回的结果中有好几组不同`job`的结果，虽然都相同，不如过滤掉冗余的好处理
- 指标的简单说明（暂时没有用）

`get_metrics_of_node`负责遍历`metric_info_dict`，然后调用`get_node_metric`提取这些指标。

