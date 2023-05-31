#
``` bash
# install
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=${CUDA_HOME}/bin:${PATH}
pip install -e.
```
``` bash

python tools/detect.py configs/condlane/resnet101_culane.py --img demo/boreas \
          --load_from modelzoo/condlane_r101_culane.pth --savedir ./vis
```