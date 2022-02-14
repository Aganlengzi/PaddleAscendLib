# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from paddle.fluid import core
from distutils.sysconfig import get_python_lib
from distutils.core import setup, Extension
from setuptools.command.build_ext import build_ext

# refer: https://note.qidong.name/2018/03/setup-warning-strict-prototypes
# Avoid a gcc warning below:
# cc1plus: warning: command line option ‘-Wstrict-prototypes’ is valid
# for C/ObjC but not for C++
class BuildExt(build_ext):
    def build_extensions(self):
        self.compiler.compiler_so.remove('-Wstrict-prototypes')
        super(BuildExt, self).build_extensions()

# cc flags
paddle_extra_compile_args = [
    '-std=c++14', '-shared', '-fPIC', '-DPADDLE_WITH_CUSTOM_KERNEL=ON'
]
# paddle include path
site_packages_path = get_python_lib()
paddle_custom_kernel_include = [
    os.path.join(site_packages_path, 'paddle', 'include'),
]
# paddle libs path
paddle_custom_kernel_library_dir = [
    os.path.join(site_packages_path, 'paddle', 'fluid'),
]
# paddle libs
libs = [':core_avx.so']
if not core.has_avx_core and core.has_noavx_core:
    libs = [':core_noavx.so']

# NPU related info:
ACL_HOME = '/usr/local/Ascend/ascend-toolkit/latest/'
ACL_INCLUDE_PATH='fwkacllib/include'
ACL_LIBRARY_PATH='fwkacllib/lib64'

if core.is_compiled_with_npu():
    paddle_extra_compile_args += [
            '-D_GLIBCXX_USE_CXX11_ABI=0',
            '-DPADDLE_WITH_ASCEND_CL=ON',
            '-Wno-parentheses',
            '-Wno-terminate',
            '-Wno-deprecated-declarations',
            ]
    paddle_custom_kernel_include += [
        ACL_HOME + ACL_INCLUDE_PATH,
        '.', './common',
        ]
    paddle_custom_kernel_library_dir += [ACL_HOME + ACL_LIBRARY_PATH,]

# Third_party deps
BOOST_PATH = '/workspace/dev/Paddle/build/third_party/boost/src/extern_boost'
GFLAGS_PATH = '/workspace/dev/Paddle/build/third_party/install/gflags/include'
GLOG_PATH = '/workspace/dev/Paddle/build/third_party/install/glog/include'
paddle_custom_kernel_include += [
    BOOST_PATH,
    GFLAGS_PATH,
    GLOG_PATH,
    ]


custom_npu_kernel_module = Extension(
    'custom_kernel_add',
    sources=['custom_elementwise_add.cc', 'npu_op_runner.cc', 'common/logging.cc'],
    include_dirs=paddle_custom_kernel_include,
    library_dirs=paddle_custom_kernel_library_dir,
    libraries=libs,
    extra_compile_args=paddle_extra_compile_args)

setup(
    name='custom_npu_kernel_lib',
    version='1.0',
    description='custom kernel fot compiling',
    cmdclass={'build_ext': BuildExt},
    ext_modules=[custom_npu_kernel_module])

