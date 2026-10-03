#!/bin/bash
set -euo pipefail
/usr/local/cuda-12.8/bin/cuobjdump --dump-resource-usage \
  /home/l/work/flash-next/prod-w48/flash_attn_v100_cuda.cpython-312-x86_64-linux-gnu.so |
  awk '/^arch = / {arch=$0}
       /^ Function .*grouped_sparse_page4_plan_kernel/ {print arch; show=2}
       show>0 {print; show--}'
