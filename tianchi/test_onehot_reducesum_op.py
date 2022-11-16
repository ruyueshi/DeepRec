
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
    for _ in range(count):
        f()
    end = time.time()
    print(title, (end - begin) / count * 1000, " ms")


class OneHotReduceSumOpsTest(test.TestCase):
    def __init__(self, method_name='run_one_hot_reduce_sum_ops_test'):
        super().__init__(methodName=method_name)
    
    def test_one_hot_reduce_sum_debug(self):
        input("pid: " + str(os.getpid()) +", press enter to continue")
        a = tf.constant([[-1,0,1],[2,3,4]])
        b = tf.one_hot(a, depth=5, axis=-1)
        c = tf.reduce_sum(b, axis=[-2])
        d = tf.one_hot_reduce_sum(a, depth=5, axis_reducesum=-2)
        with tf.Session() as sess:
            # print("a:", sess.run(a))
            # print("b:", sess.run(b))
            # print("c:", sess.run(c))
            # print("d:", sess.run(d))
            self.assertAllEqual(c, d)

    def test_one_hot_reduce_sum(self):
        depth = 10000
        a = tf.constant(np.random.randint(0, depth, [512, 50]))
        fused = tf.one_hot_reduce_sum(a, depth=depth, axis_onehot=-1, axis_reducesum=-2)  # axis_onehot default is -1 and axis_reducesum default is -2
        gt = tf.reduce_sum(tf.one_hot(a, depth=depth), axis=-2)
        count = 10
        with self.session() as sess:
            t0 = time.time()
            res_fused = sess.run(fused)
            t1 = time.time()
            res_gt = sess.run(gt)
            t2 = time.time()
            print("=== run warmup:\n\tfused op:{} s\n\torigin op:{} s".format(t1 - t0, t2 - t1))
            self.assertAllEqual(res_fused, res_gt)
            print("=== run multiple times:")
            timeit("\tfused op:", lambda: sess.run(fused), count)
            timeit("\torigin op:", lambda: sess.run(gt), count)


if __name__ == "__main__":
    test.main()

