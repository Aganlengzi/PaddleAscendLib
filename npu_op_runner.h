/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/extension.h"

#include "boost/config.hpp"
#include "boost/variant.hpp"

#include "acl/acl.h"

using Tensor = paddle::Tensor;
using NPUAttribute =
    boost::variant<boost::blank, int, float, std::string, std::vector<int>,
                   std::vector<float>, std::vector<std::string>, bool,
                   std::vector<bool>, int64_t, std::vector<int64_t>,
                   std::vector<double>, std::vector<std::vector<int64_t>>>;

using NPUAttributeMap = std::unordered_map<std::string, NPUAttribute>;

class NpuOpRunner {
 public:
  NpuOpRunner();
  explicit NpuOpRunner(const std::string &op_type);
  NpuOpRunner(const std::string &op_type,
              const std::vector<Tensor> &inputs = {},
              const std::vector<Tensor> &outputs = {},
              const NPUAttributeMap &attrs = {});

  NpuOpRunner(const NpuOpRunner &runner) = delete;
  NpuOpRunner &operator=(const NpuOpRunner &runner) = delete;

  ~NpuOpRunner();

  const std::string &Type();

  NpuOpRunner &SetType(const std::string &name);

  NpuOpRunner &AddAttr(const std::string &name, const NPUAttribute &attr);

  NpuOpRunner &AddAttrDataType(const std::string &name, const NPUAttribute &attr);

  NpuOpRunner &AddAttrs(const NPUAttributeMap &attrs);

  NpuOpRunner &AddInput(const Tensor &tensor);

  NpuOpRunner &AddInput(const Tensor &tensor, aclMemType mem_type);

  NpuOpRunner &AddInput(std::vector<int32_t> &&dims);

  NpuOpRunner &AddInput(std::vector<int64_t> &&dims);

  NpuOpRunner &AddInput(std::vector<float> &&values);

  NpuOpRunner &AddInput(std::vector<double> &&values);

  NpuOpRunner &AddOutput(const Tensor &tensor);

  NpuOpRunner &AddInputs(const std::vector<Tensor> &tensors);

  NpuOpRunner &AddInputNames(const std::vector<std::string> &names);

  NpuOpRunner &AddOutputs(const std::vector<Tensor> &tensors);

  aclTensorDesc *GetInputDesc(size_t index);

  aclTensorDesc *GetOutputDesc(size_t index);

  std::vector<aclTensorDesc *> &GetInputDescs();

  std::vector<aclTensorDesc *> &GetOutputDescs();

  std::vector<aclDataBuffer *> &GetInputBuffers();

  std::vector<aclDataBuffer *> &GetOutputBuffers();

  void Run(aclrtStream stream = nullptr) const;

 private:
  aclTensorDesc *CreateTensorDesc(Tensor tensor,
                                  aclMemType mem_type = ACL_MEMTYPE_DEVICE);
  aclDataBuffer *CreateDataBuffer(Tensor tensor);

 private:
  std::string op_type_;
  std::vector<aclDataBuffer *> input_buffers_;
  std::vector<aclDataBuffer *> output_buffers_;
  std::vector<aclTensorDesc *> input_descs_;
  std::vector<aclTensorDesc *> output_descs_;
  std::vector<Tensor> host_tensors_;
  aclopAttr *attr_{nullptr};
};
