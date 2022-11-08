/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// See docs in ../ops/array_ops.cc

#define EIGEN_USE_THREADS

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/kernels/one_hot_reduce_sum_op.h"

#include <time.h>
#include <memory>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/overflow.h"
#include "tensorflow/core/util/work_sharder.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T, typename TI>
class OneHotReduceSumOp : public OpKernel {
 public:
  explicit OneHotReduceSumOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("axis_onehot", &axis_onehot_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("axis_reducesum", &axis_reducesum_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& indices = ctx->input(0);
    const Tensor& depth = ctx->input(1);
    const Tensor& on_value = ctx->input(2);
    const Tensor& off_value = ctx->input(3);
    const TensorShape& indices_shape = indices.shape();

    const int indices_dims = indices_shape.dims();
    const int one_hot_dims = indices_dims + 1;
    const int output_dims = indices_dims;
    const int indices_num = indices.NumElements();

    // Preliminary validation of sizes.
    OP_REQUIRES(
        ctx, axis_onehot_ == -1,  // || (axis_onehot_ >= 0 && axis_onehot_ <= indices_dims),
        errors::InvalidArgument("Expected axis to be -1 or between [0, ",
                                output_dims, ").  But received: ", axis_onehot_));
    OP_REQUIRES(ctx, axis_reducesum_ == -2,
        errors::InvalidArgument("Expected axis to be -2.  But received: ", axis_reducesum_));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(depth.shape()),
                errors::InvalidArgument("depth must be a scalar, but got: ",
                                        depth.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(on_value.shape()),
                errors::InvalidArgument("on_value must be a scalar, but got: ",
                                        on_value.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(off_value.shape()),
                errors::InvalidArgument("off_value must be a scalar, but got: ",
                                        off_value.shape().DebugString()));

    const int axis_onehot = (axis_onehot_ == -1) ? indices_dims : axis_onehot_;
    const int axis_reducesum = (one_hot_dims + axis_reducesum_ % one_hot_dims) % one_hot_dims;

    // The one-hot dimension.
    const int32 depth_v = depth.scalar<int32>()();
    OP_REQUIRES(
        ctx, depth_v >= 0,
        errors::InvalidArgument("depth must be non-negative, got: ", depth_v));
    OP_REQUIRES(
        ctx,
        MultiplyWithoutOverflow(indices_num, depth_v) >= 0,
        errors::InvalidArgument("OneHotReduceSumOp result would have shape ",
                                indices_shape.DebugString(), " + [", depth_v,
                                "], which exceeds 2**63 - 1 elements"));

    TensorShape output_shape = indices_shape;
    output_shape.InsertDim(axis_onehot, depth_v);
    output_shape.RemoveDim(axis_reducesum);

    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    const int output_num = output->NumElements();

    TI* indices_ptr = static_cast<TI*>(indices.data());
    T* output_ptr = static_cast<T*>(output->data());
    T* on_value_ptr = static_cast<T*>(on_value.data());
    T* off_value_ptr = static_cast<T*>(off_value.data());

    if (output_shape.num_elements() > 0) {
      // prefix_dim_size == # of elements before the axis_onehot
      // depth_v == # of elements per axis_onehot
      // suffix_dim_size == # of elements after the axis_onehot
      int64 prefix_dim_size = 1;
      for (int i = 0; i < axis_onehot - 1; ++i) {
        prefix_dim_size *= indices_shape.dim_size(i);
      }
      int64 suffix_dim_size = indices_num / prefix_dim_size;

      auto work = [this, &indices_ptr, &output_ptr, &on_value_ptr, &off_value_ptr, &depth_v, &suffix_dim_size, &output_num](int64 start, int64 end) {
        for (int64 i = start; i < end; i++) {
          auto val = indices_ptr[i];
          if (val < 0 || val >= depth_v) {
            continue;
          }
          int64 row = i / suffix_dim_size;
          output_ptr[row * depth_v + val] += (*on_value_ptr);
        }
      };

      std::memset(output_ptr, static_cast<int>(*off_value_ptr), output->NumElements() * sizeof(T));
      if (ctx) {
        auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
        VLOG(1) << "tf shard, num_threads = " << worker_threads.num_threads;
        Shard(worker_threads.num_threads, worker_threads.workers, indices_num, 5000, work);
      } else {
        VLOG(1) << "serial";
        work(0, indices_num);
      }
    }
  }

 private:
  int32 axis_onehot_;
  int32 axis_reducesum_;

  TF_DISALLOW_COPY_AND_ASSIGN(OneHotReduceSumOp);
};

#define REGISTER_ONE_HOT_INDEX(type, index_type)                \
  REGISTER_KERNEL_BUILDER(Name("OneHotReduceSum")               \
                              .Device(DEVICE_CPU)               \
                              .TypeConstraint<index_type>("TI") \
                              .TypeConstraint<type>("T")        \
                              .HostMemory("depth"),             \
                          OneHotReduceSumOp<CPUDevice, type, index_type>);

#define REGISTER_ONE_HOT(type)         \
  REGISTER_ONE_HOT_INDEX(type, uint8); \
  REGISTER_ONE_HOT_INDEX(type, int32); \
  REGISTER_ONE_HOT_INDEX(type, int64)

// TF_CALL_ALL_TYPES(REGISTER_ONE_HOT);
TF_CALL_float(REGISTER_ONE_HOT);
TF_CALL_double(REGISTER_ONE_HOT);

// #if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
//     (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)

// // Forward declarations of the functor specializations for GPU.
// namespace functor {
// #define DECLARE_GPU_SPEC_INDEX(T, TI)                                      \
//   template <>                                                              \
//   void OneHotReduceSumOp<GPUDevice, T, TI>::Compute( \
//       const GPUDevice& d, const typename TTypes<TI>::ConstMatrix& indices, \
//       const typename TTypes<T>::ConstScalar& on_value,                     \
//       const typename TTypes<T>::ConstScalar& off_value,                    \
//       typename TTypes<T, 3>::Tensor* output);                              \
//   extern template struct OneHotReduceSumOp<GPUDevice, T, TI>;

// #define DECLARE_GPU_SPEC(T)         \
//   DECLARE_GPU_SPEC_INDEX(T, uint8); \
//   DECLARE_GPU_SPEC_INDEX(T, int32); \
//   DECLARE_GPU_SPEC_INDEX(T, int64);

// TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPEC);
// TF_CALL_bool(DECLARE_GPU_SPEC);
// TF_CALL_int32(DECLARE_GPU_SPEC);
// TF_CALL_int64(DECLARE_GPU_SPEC);

// #undef DECLARE_GPU_SPEC_INDEX
// #undef DECLARE_GPU_SPEC

// }  // namespace functor

// // Registration of the GPU implementations.
// #define REGISTER_ONE_HOT_GPU_INDEX(type, index_type)            \
//   REGISTER_KERNEL_BUILDER(Name("OneHotReduceSumOp")                        \
//                               .Device(DEVICE_GPU)               \
//                               .TypeConstraint<index_type>("TI") \
//                               .TypeConstraint<type>("T")        \
//                               .HostMemory("depth"),             \
//                           OneHotOp<GPUDevice, type, index_type>);

// #define REGISTER_ONE_HOT_GPU(type)         \
//   REGISTER_ONE_HOT_GPU_INDEX(type, uint8); \
//   REGISTER_ONE_HOT_GPU_INDEX(type, int32); \
//   REGISTER_ONE_HOT_GPU_INDEX(type, int64);

// TF_CALL_GPU_NUMBER_TYPES(REGISTER_ONE_HOT_GPU);
// TF_CALL_bool(REGISTER_ONE_HOT_GPU);
// TF_CALL_int32(REGISTER_ONE_HOT_GPU);
// TF_CALL_int64(REGISTER_ONE_HOT_GPU);

// #undef REGISTER_ONE_HOT_GPU_INDEX
// #undef REGISTER_ONE_HOT_GPU

// #endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
