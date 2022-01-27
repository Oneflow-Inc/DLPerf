import numpy as np

import paddle
from plsc import Entry

NUM_EPOCHES = 1
LOSS_TYPE = 'dist_arcface'

NUM_SAMPLES = 5822653
NUM_CLASSES = 85742


def arc_train(*args):
    def reader():
        for i in range(NUM_SAMPLES):
            yield np.random.normal(size=(3, 112, 112)), int(np.random.randint(NUM_CLASSES))

    def mapper(x):
        return x

    THREAD=8
    BUF_SIZE=5000

    return paddle.reader.xmap_readers(mapper, reader, THREAD, BUF_SIZE)



def main():
    ins = Entry()
    ins.set_model_save_dir("./checkpoints")
    ins.set_train_epochs(NUM_EPOCHES)
    ins.set_loss_type(LOSS_TYPE)
    ins.set_train_batch_size=128

    ins.set_with_test(False)
    ins.set_class_num(NUM_CLASSES)
    ins.set_log_period(10)

    ins.train_reader = arc_train()
    ins.train()


if __name__ == "__main__":
    main()
