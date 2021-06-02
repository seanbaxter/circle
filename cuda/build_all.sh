set -x
circle --cuda-path=/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda/11.2/ -sm_52 -sm_75 hello.cu -lcudart -o hello
circle --cuda-path=/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda/11.2/ -sm_35 -sm_52 -sm_61 -sm_75 targets.cxx
circle --cuda-path=/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda/11.2/ -sm_35 -sm_52 -sm_61 -sm_75 if_target.cxx -lcudart
circle --cuda-path=/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda/11.2/ -sm_35 -sm_52 -sm_61 -sm_75 launch.cxx -lcudart
circle --cuda-path=/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda/11.2/ -sm_35 -sm_52 -sm_61 -sm_75 bad_launch.cxx -lcudart -g