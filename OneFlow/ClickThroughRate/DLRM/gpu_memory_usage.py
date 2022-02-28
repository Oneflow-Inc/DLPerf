import time
from pynvml import *

nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)
running = True

mem_threshold = 32*1024*1024
state = 'init' #'Detecting'

device0_max_used_mem = 0
while running == True:
    time.sleep(1)
    info = nvmlDeviceGetMemoryInfo(handle)
    if state == 'init':
        if info.used > mem_threshold:
            state = 'Detecting'
    elif state == 'Detecting':
        if info.used < mem_threshold:
            running = False
        else:
            device0_max_used_mem = max(device0_max_used_mem, info.used)

nvmlShutdown()
print('max device0 memory usage is:', device0_max_used_mem)
