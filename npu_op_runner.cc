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

#include "npu_op_runner.h"

#include <map>
#include "acl/acl_op_compiler.h"
#include "common/logging.h"

static std::map<paddle::experimental::DataType, aclDataType>
    DTYPE_2_ACL_DTYPE = {
        {paddle::experimental::DataType::BOOL, ACL_BOOL},
        {paddle::experimental::DataType::UINT8, ACL_UINT8},
        {paddle::experimental::DataType::INT8, ACL_INT8},
        {paddle::experimental::DataType::INT16, ACL_INT16},
        {paddle::experimental::DataType::INT32, ACL_INT32},
        {paddle::experimental::DataType::INT64, ACL_INT64},
        {paddle::experimental::DataType::FLOAT16, ACL_FLOAT16},
        {paddle::experimental::DataType::FLOAT32, ACL_FLOAT},
        {paddle::experimental::DataType::FLOAT64, ACL_DOUBLE},
};

static std::map<paddle::experimental::DataLayout, aclFormat> DATA_LAYOUT_2_ACL_FORMAT = {
    {paddle::experimental::DataLayout::NCHW, ACL_FORMAT_NCHW},
    {paddle::experimental::DataLayout::NHWC, ACL_FORMAT_NHWC},
    {paddle::experimental::DataLayout::ANY, ACL_FORMAT_ND},
};

aclDataType ConvertToNpuDtype(paddle::experimental::DataType dtype) {
  auto iter = DTYPE_2_ACL_DTYPE.find(dtype);
  PD_CHECK(iter != DTYPE_2_ACL_DTYPE.end());
  return iter->second;
}

aclFormat ConvertToNpuFormat(paddle::experimental::DataLayout layout) {
  auto iter = DATA_LAYOUT_2_ACL_FORMAT.find(layout);
  PD_CHECK(iter != DATA_LAYOUT_2_ACL_FORMAT.end());
  return iter->second;
}

NpuOpRunner::NpuOpRunner() {}

NpuOpRunner::NpuOpRunner(const std::string &op_type) : op_type_(op_type) {}

NpuOpRunner::NpuOpRunner(const std::string &op_type,
                         const std::vector<Tensor> &inputs,
                         const std::vector<Tensor> &outputs,
                         const NPUAttributeMap &attrs) : op_type_(op_type) {
  AddInputs(inputs);
  AddOutputs(outputs);
  AddAttrs(attrs);
}

NpuOpRunner::~NpuOpRunner() {
  VLOG(4) << "Free NpuOpRunner(" << this << ") of " << op_type_;
  // Is it safe to free the descs/buffers after run called in host ?
  aclopDestroyAttr(attr_);  // return void
  for (auto desc : input_descs_) {
    aclDestroyTensorDesc(desc);
  }
  for (auto desc : output_descs_) {
    aclDestroyTensorDesc(desc);
  }
  for (auto buffer : input_buffers_) {
    PD_CHECK(ACL_ERROR_NONE == aclDestroyDataBuffer(buffer));
  }
  for (auto buffer : output_buffers_) {
    PD_CHECK(ACL_ERROR_NONE == aclDestroyDataBuffer(buffer));
  }
}

const std::string &NpuOpRunner::Type() { return op_type_; }

NpuOpRunner &NpuOpRunner::SetType(const std::string &name) {
  op_type_ = name;
  return *this;
}

NpuOpRunner &NpuOpRunner::AddAttr(const std::string &name,
                                  const NPUAttribute &attr) {
  if (!attr_) {
    attr_ = aclopCreateAttr();
  }
  if (attr.type() == typeid(bool)) {
    PD_CHECK(ACL_ERROR_NONE ==
        aclopSetAttrBool(attr_, name.c_str(), boost::get<bool>(attr)));
  } else if (attr.type() == typeid(int)) {
    PD_CHECK(ACL_ERROR_NONE ==
        aclopSetAttrInt(attr_, name.c_str(), boost::get<int>(attr)));

  } else if (attr.type() == typeid(int64_t)) {
    PD_CHECK(ACL_ERROR_NONE ==
        aclopSetAttrInt(attr_, name.c_str(), boost::get<int64_t>(attr)));
  } else if (attr.type() == typeid(float)) {
    PD_CHECK(ACL_ERROR_NONE ==
        aclopSetAttrFloat(attr_, name.c_str(), boost::get<float>(attr)));
  } else if (attr.type() == typeid(std::vector<bool>)) {
    auto a = boost::get<std::vector<bool>>(attr);
    std::vector<uint8_t> cast_a;
    for (auto it : a) {
      cast_a.push_back(static_cast<uint8_t>(it));
    }
    PD_CHECK(ACL_ERROR_NONE ==aclopSetAttrListBool(
        attr_, name.c_str(), cast_a.size(), cast_a.data()));
  } else if (attr.type() == typeid(std::vector<int>)) {
    auto a = boost::get<std::vector<int>>(attr);
    std::vector<int64_t> cast_a;
    for (auto it : a) {
      cast_a.push_back(static_cast<int64_t>(it));
    }
    PD_CHECK(ACL_ERROR_NONE == 
        aclopSetAttrListInt(attr_, name.c_str(), cast_a.size(), cast_a.data()));
  } else if (attr.type() == typeid(std::vector<int64_t>)) {
    auto a = boost::get<std::vector<int64_t>>(attr);
    PD_CHECK(ACL_ERROR_NONE == 
        aclopSetAttrListInt(attr_, name.c_str(), a.size(), a.data()));
  } else if (attr.type() == typeid(std::vector<float>)) {
    auto a = boost::get<std::vector<float>>(attr);
    PD_CHECK(ACL_ERROR_NONE == 
        aclopSetAttrListFloat(attr_, name.c_str(), a.size(), a.data()));
  } else if (attr.type() == typeid(std::string)) {
    auto a = boost::get<std::string>(attr);
    PD_CHECK(ACL_ERROR_NONE == 
        aclopSetAttrString(attr_, name.c_str(), a.c_str()));
  } else if (attr.type() == typeid(std::vector<std::string>)) {
    auto a = boost::get<std::vector<std::string>>(attr);
    std::vector<const char *> s;
    for (auto &it : a) {
      s.push_back(it.data());
    }
    PD_CHECK(ACL_ERROR_NONE == 
        aclopSetAttrListString(attr_, name.c_str(), s.size(), s.data()));
  } else if (attr.type() == typeid(std::vector<std::vector<int64_t>>)) {
    auto a = boost::get<std::vector<std::vector<int64_t>>>(attr);
    std::vector<int64_t *> data;
    std::vector<int> num;
    for (auto &&v : a) {
      data.push_back(v.data());
      num.push_back(v.size());
    }
    PD_CHECK(ACL_ERROR_NONE == aclopSetAttrListListInt(
        attr_, name.c_str(), data.size(), num.data(), data.data()));
  } else {
    PD_CHECK(false,
        "Can not convert attribubte '%s' to convert to aclopAttr", name);
  }
  return *this;
}

NpuOpRunner &NpuOpRunner::AddAttrDataType(const std::string &name,
                                          const NPUAttribute &attr) {
  PD_CHECK(attr.type() == typeid(int));
  if (!attr_) {
    attr_ = aclopCreateAttr();
  }
  VLOG(4) << "AddAttrDataType call";
  auto dtype = ConvertToNpuDtype(
      static_cast<paddle::experimental::DataType>(boost::get<int>(attr)));
  PD_CHECK(ACL_ERROR_NONE == aclopSetAttrDataType(attr_, name.c_str(), dtype));
  return *this;
}

NpuOpRunner &NpuOpRunner::AddAttrs(const NPUAttributeMap &attrs) {
  for (const auto &pair : attrs) {
    AddAttr(pair.first, pair.second);
  }
  return *this;
}

NpuOpRunner &NpuOpRunner::AddInput(const Tensor &tensor) {
  // create aclTensorDesc
  input_descs_.emplace_back(CreateTensorDesc(tensor));
  // create aclDataBuffer
  input_buffers_.emplace_back(CreateDataBuffer(tensor));
  return *this;
}

NpuOpRunner &NpuOpRunner::AddInput(const Tensor &tensor, aclMemType mem_type) {
  // create aclTensorDesc
  input_descs_.emplace_back(CreateTensorDesc(tensor, mem_type));
  // create aclDataBuffer
  input_buffers_.emplace_back(CreateDataBuffer(tensor));
  return *this;
}

NpuOpRunner &NpuOpRunner::AddOutput(const Tensor &tensor) {
  // create aclTensorDesc
  output_descs_.emplace_back(CreateTensorDesc(tensor));
  // create aclDataBuffer
  output_buffers_.emplace_back(CreateDataBuffer(tensor));
  return *this;
}

NpuOpRunner &NpuOpRunner::AddInputs(const std::vector<Tensor> &tensors) {
  input_descs_.reserve(tensors.size());
  input_buffers_.reserve(tensors.size());
  for (auto tensor : tensors) {
    // create aclTensorDesc
    input_descs_.emplace_back(CreateTensorDesc(tensor));
    // create aclDataBuffer
    input_buffers_.emplace_back(CreateDataBuffer(tensor));
  }
  return *this;
}

// NOTE(zhiqiu): For operators whose input is a list (such as concat, stack),
// It is needed to set the name of each input tensor.
NpuOpRunner &NpuOpRunner::AddInputNames(const std::vector<std::string> &names) {
  PD_CHECK(names.size() == input_descs_.size());
  for (size_t i = 0; i < names.size(); ++i) {
    aclSetTensorDescName(input_descs_[i], names[i].c_str());
  }
  return *this;
}

NpuOpRunner &NpuOpRunner::AddOutputs(const std::vector<Tensor> &tensors) {
  output_descs_.reserve(tensors.size());
  output_buffers_.reserve(tensors.size());
  for (auto tensor : tensors) {
    // create aclTensorDesc
    output_descs_.emplace_back(CreateTensorDesc(tensor));
    // create aclDataBuffer
    output_buffers_.emplace_back(CreateDataBuffer(tensor));
  }
  return *this;
}

aclTensorDesc *NpuOpRunner::GetInputDesc(size_t index) {
  PD_CHECK(index < input_descs_.size());
  return input_descs_[index];
}

aclTensorDesc *NpuOpRunner::GetOutputDesc(size_t index) {
  PD_CHECK(index < output_descs_.size());
  return output_descs_[index];
}

std::vector<aclTensorDesc *> &NpuOpRunner::GetInputDescs() {
  return input_descs_;
}

std::vector<aclTensorDesc *> &NpuOpRunner::GetOutputDescs() {
  return output_descs_;
}

std::vector<aclDataBuffer *> &NpuOpRunner::GetInputBuffers() {
  return input_buffers_;
}

std::vector<aclDataBuffer *> &NpuOpRunner::GetOutputBuffers() {
  return output_buffers_;
}

aclTensorDesc *NpuOpRunner::CreateTensorDesc(Tensor tensor,
                                             aclMemType mem_type) {
  auto dtype = ConvertToNpuDtype(tensor.type());
  auto format = ConvertToNpuFormat(tensor.layout());
  auto dims = tensor.shape();
  int size = dims.size();

  if (op_type_ == "DropOutGenMask" && size == 1 && *(dims.data()) == 1) {
    size = 0;
  }

  VLOG(4) << "NPU dtype:" << dtype << " "
          << "rank:" << dims.size() << " dims: TBD" /* << tensor.dims()*/
          << " format:" << format;

  auto *desc = aclCreateTensorDesc(dtype, size, dims.data(), format);
  PD_CHECK(desc != NULL);
  PD_CHECK(ACL_ERROR_NONE == aclSetTensorStorageFormat(desc, format));
  PD_CHECK(ACL_ERROR_NONE == aclSetTensorStorageShape(desc, size, dims.data()));
  if (mem_type == ACL_MEMTYPE_HOST) {
    PD_CHECK(ACL_ERROR_NONE == aclSetTensorPlaceMent(desc, mem_type));
  }
  return desc;
}

aclDataBuffer *NpuOpRunner::CreateDataBuffer(Tensor tensor) {
  void *ptr = tensor.mutable_data<int8_t>();
  size_t tensor_memory_size = tensor.size() * sizeof(tensor.dtype());
  VLOG(4) << "NPU ptr: " << ptr << ", size: " << tensor_memory_size;    // NOT provided tensor.memory_size()
  auto *buffer = aclCreateDataBuffer(ptr, tensor_memory_size);
  PD_CHECK(buffer != NULL);
  return buffer;
}

void NpuOpRunner::Run(aclrtStream stream) const {
  PD_CHECK(stream != NULL);

  VLOG(5) << "NpuOpRunner(" << this << ") Run:";
  VLOG(4) << "op_type: " << op_type_;
  VLOG(4) << "input_desc.size: " << input_descs_.size();
  VLOG(4) << "output_desc.size: " << output_descs_.size();
  VLOG(4) << "attr: " << attr_;
  VLOG(4) << "stream: " << stream;

  aclError ret = aclopCompileAndExecute(
      op_type_.c_str(), input_descs_.size(), input_descs_.data(),
      input_buffers_.data(), output_descs_.size(), output_descs_.data(),
      output_buffers_.data(), attr_, ACL_ENGINE_SYS, ACL_COMPILE_SYS, NULL,
      stream);
  VLOG(4) << "after aclopCompileAndExecute: " << ret;
  PD_CHECK(ACL_ERROR_NONE == ret);
}
