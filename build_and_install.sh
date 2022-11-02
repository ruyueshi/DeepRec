set -e
# build
bazel build --config=opt -c opt --config=mkl_threadpool --define build_with_mkl_dnn_v1_only=true //tensorflow/tools/pip_package:build_pip_package
# install
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg/
pip uninstall -y tensorflow
pip install /tmp/tensorflow_pkg/tensorflow-1.15.5+deeprec2206-cp36-cp36m-linux_x86_64.whl
set +e

