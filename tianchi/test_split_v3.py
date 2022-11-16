
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.platform import test


def timeit(title, f, count=10):
    if count == 0:
        return
    f()  # warmup
    begin = time.time()
    #for _ in range(count):
    while True:
        f()
    end = time.time()
    print(title, (end - begin) / count * 1000, " ms")


class OneStringSplitOpsTest(test.TestCase):
    def __init__(self, method_name='run_one_hot_reduce_sum_ops_test'):
        super().__init__(methodName=method_name)
    
    def test_string_split(self):
        # input("pid: " + str(os.getpid()) +", press enter to continue")
        a = tf.strings.split(["14352;14909;14884;14868;0;28578;13599;0;0;14344;14909;0;0;14720;30821;21919;0;0;0;11384;11384;11384;11384;0;24326;30688;42609;37907;0;0;35563;26832;19507;24326;6322;6322;0;0;37776;38099;18186;0;26017;11427"], 
            sep=";", maxsplit=-2)
        b = tf.strings.split(["14352;14909;14884;14868;0;28578;13599;0;0;14344;14909;0;0;14720;30821;21919;0;0;0;11384;11384;11384;11384;0;24326;30688;42609;37907;0;0;35563;26832;19507;24326;6322;6322;0;0;37776;38099;18186;0;26017;11427"], 
            sep=";", maxsplit=-1)
        with self.session() as sess:
            # print("a:", sess.run(a))
            gt = sess.run(a)
            pred = sess.run(b)
            self.assertAllEqual(gt[0], pred[0])
            self.assertAllEqual(gt[1], pred[1])
            self.assertAllEqual(gt[2], pred[2])
            #timeit("splitv2:", lambda: sess.run(a), 10)
            timeit("splitv3:", lambda: sess.run(b), 10)

if __name__ == "__main__":
    test.main()

