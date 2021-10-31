set -x

# Hello
circle --cuda-path=/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda/11.2/ -sm_52 -sm_75 hello.cu -o hello

# Standard library examples.
circle --cuda-path=/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda/11.2/ -sm_52 roman.cxx -o roman
circle --cuda-path=/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda/11.2/ -sm_52 unique.cxx -o unique
circle --cuda-path=/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda/11.2/ -sm_52 qsort.cxx -o qsort
circle --cuda-path=/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda/11.2/ -sm_52 virtual.cxx -o virtual
circle --cuda-path=/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda/11.2/ -sm_52 any.cxx -o any
circle storage_only.cxx
circle --cuda-path=/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda/11.2/ -sm_52 vec.cxx -o vec

# Targets
circle --cuda-path=/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda/11.2/ -sm_35 -sm_52 -sm_61 -sm_75 targets.cxx
circle --cuda-path=/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda/11.2/ -sm_35 -sm_52 -sm_61 -sm_75 if_target.cxx 
circle --cuda-path=/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda/11.2/ -sm_35 -sm_52 -sm_61 -sm_75 launch.cxx 
circle --cuda-path=/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda/11.2/ -sm_35 -sm_52 -sm_61 -sm_75 tuning1.cxx 
circle --cuda-path=/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda/11.2/ -sm_35 -sm_52 -sm_61 -sm_75 launch2.cxx 
circle --cuda-path=/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda/11.2/ -sm_35 -sm_52 -sm_61 -sm_75 bad_launch.cxx -g