#!/bin/bash

USE_GPU_INFERENCE=0 python3 label_image.py -m models/mobilenet_ssd_v2_coco_quant_postprocess.tflite -l labels/coco.txt -e /usr/lib/libvx_delegate.so
