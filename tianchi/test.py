
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
        a = tf.constant([[0,1,1],[2,3,3]])
        b = tf.one_hot(a, depth=5, axis=-1)
        c = tf.reduce_sum(b, axis=[-2])
        d = tf.one_hot_reduce_sum(a, depth=5, axis_reducesum=-2)
        with tf.Session() as sess:
            print("a:", sess.run(a))
            print("b:", sess.run(b))
            print("c:", sess.run(c))
            print("d:", sess.run(d))
            self.assertAllEqual(c, d)

    def test_one_hot_reduce_sum_acc(self):
        """
        经过测试发现：同时运行 origin op 和 fused op 会导致 fused op 的速度测出来慢很多。
        要想测试 origin op 和 fused op 的真实速度，可以运行 test_one_hot_reduce_sum_perf_gt 和 
        test_one_hot_reduce_sum_perf_fused 这两个单测。
        """
        depth = 100
        a = tf.constant(np.random.randint(0, depth, [200, 300, 400]))
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
    
    def test_one_hot_reduce_sum_perf_fused(self):
        depth = 200
        a = tf.constant(np.random.randint(0, depth, [100, 100, 4000]))
        fused = tf.one_hot_reduce_sum(a, depth=depth, axis_onehot=-1, axis_reducesum=5000)  # axis_onehot default is -1 and axis_reducesum default is -2
        count = 5
        with self.session() as sess:
            t0 = time.time()
            sess.run(fused)
            t1 = time.time()
            print("=== run warmup:\n\tfused op:{} s".format(t1 - t0))
            print("=== run multiple times:")
            timeit("\tfused op:", lambda: sess.run(fused), count)

    def test_one_hot_reduce_sum_perf_gt(self):
        depth = 200
        a = tf.constant(np.random.randint(0, depth, [100, 100, 4000]))
        gt = tf.reduce_sum(tf.one_hot(a, depth=depth), axis=-2)
        count = 5
        with self.session() as sess:
            t0 = time.time()
            sess.run(gt)
            t1 = time.time()
            print("=== run warmup:\n\torigin op:{} s".format(t1 - t0))
            print("=== run multiple times:")
            timeit("\torigin op:", lambda: sess.run(gt), count)


if __name__ == "__main__":
    test.main()

