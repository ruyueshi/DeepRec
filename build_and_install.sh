set -e
# build
# release version
bazel build -s --config=opt -c opt --config=mkl_threadpool --define build_with_mkl_dnn_v1_only=true //tensorflow/tools/pip_package:build_pip_package
# # debug version
# bazel build -s -c dbg --cxxopt='-g' --config=mkl_threadpool --define build_with_mkl_dnn_v1_only=true //tensorflow/tools/pip_package:build_pip_package

# # install initally
# ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg/
# pip uninstall -y tensorflow
# pip install /tmp/tensorflow_pkg/tensorflow-1.15.5+deeprec2206-cp36-cp36m-linux_x86_64.whl

# just update so
cp bazel-bin/tensorflow/python/_pywrap_tensorflow_internal.so /home/pai/lib/python3.6/site-packages/tensorflow_core/python/

set +e

