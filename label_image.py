# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""label_image for tflite."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time

import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
import cv2
import socket
import struct
import math

def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-i',
      '--image',
      default='grace_hopper.bmp',
      help='image to be classified')
  parser.add_argument(
      '-m',
      '--model_file',
      default='mobilenet_v1_1.0_224_quant.tflite',
      help='.tflite model to be executed')
  parser.add_argument(
      '-l',
      '--label_file',
      default='labels.txt',
      help='name of file containing labels')
  parser.add_argument(
      '--input_mean',
      default=127.5, type=float,
      help='input_mean')
  parser.add_argument(
      '--input_std',
      default=127.5, type=float,
      help='input standard deviation')
  parser.add_argument(
      '--num_threads', default=None, type=int, help='number of threads')
  parser.add_argument(
      '-e',
      '--ext_delegate',
      help='external_delegate_library path')
  parser.add_argument(
      '-o',
      '--ext_delegate_options',
      help='external delegate options, format: "option1: value1; option2: value2"')

  args = parser.parse_args()

  ext_delegate = None
  ext_delegate_options = {}

  # parse extenal delegate options
  if args.ext_delegate_options is not None:
    options = args.ext_delegate_options.split(';')
    for o in options:
      kv = o.split(':')
      if(len(kv) == 2):
        ext_delegate_options[kv[0].strip()] = kv[1].strip()

  # load external delegate
  if args.ext_delegate is not None:
    print("Loading external delegate from {} with args: {}".format(args.ext_delegate, ext_delegate_options))
    ext_delegate = [ tflite.load_delegate(args.ext_delegate, ext_delegate_options) ]

  interpreter = tflite.Interpreter(
        model_path=args.model_file, experimental_delegates=ext_delegate, num_threads=args.num_threads)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # NxHxWxC, H:1, W:2
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]
  print("height: " + str(height))
  print("width: " + str(width))
  # PIL image open
  #img = Image.open(args.image).resize((width, height))
  cap = cv2.VideoCapture('v4l2src device=/dev/video4 ! video/x-raw,framerate=30/1,width=640,height=480 ! appsink', cv2.CAP_GSTREAMER)
  #cap = cv2.VideoCapture('/home/user/ml_test/images/video3.mp4')
  ret,frame = cap.read()
  frame_width = frame.shape[1]
  frame_height = frame.shape[0]
  color_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  img = Image.fromarray(color_converted).resize((width,height))

  #print(np.asarray(img))

  # add N dim
  input_data = np.expand_dims(img, axis=0)

  interpreter.set_tensor(input_details[0]['index'], input_data)

  # ignore the 1st invoke
  startTime = time.time()
  interpreter.invoke()
  delta = time.time() - startTime
  print("Warm-up time:", '%.1f' % (delta * 1000), "ms\n")

  i = 0
  while(True):
    #i = i + 1
    ret,frame = cap.read()
    color_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(color_converted).resize((width,height))

    # add N dim
    input_data = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)

    startTime = time.time()
    interpreter.invoke()
    delta = time.time() - startTime
    print("Inference time:", '%.1f' % (delta * 1000), "ms\n")

    bounding_boxes = np.squeeze(interpreter.get_tensor(252))
    classes = np.squeeze(interpreter.get_tensor(253))
    scores = np.squeeze(interpreter.get_tensor(254))
    number_of_boxes = np.squeeze(interpreter.get_tensor(255))
    confident_scores = 0

    for i in range(len(scores)):
      if(scores[i] > 0.3):
        confident_scores = confident_scores + 1

    labels = load_labels(args.label_file)
    class_names = []

    #print(len(labels))

    for i in range(len(classes)):
      class_no = int(classes[i])
      #print(class_no)
      label = labels[class_no]
      class_names.append(label)

    bounding_boxes = bounding_boxes * 255
    bounding_boxes = bounding_boxes.astype(int)
    print("------------------------------")
    #print("Bounding boxes: \n" + str(bounding_boxes[:confident_scores]) + "\n")
    print("Classes: \n" + str(class_names[:confident_scores]) + "\n")
    print("Scores: \n" + str(scores[:confident_scores]) + "\n")
    print("Number of Boxes: \n" + str(confident_scores) + "\n")
  
    for i in range(confident_scores):
      top = bounding_boxes[i][0]
      left = bounding_boxes[i][1]
      bottom = bounding_boxes[i][2]
      right = bounding_boxes[i][3]
      w = int((right-left)*(frame_width/255))
      h = int((bottom-top)*(frame_height/255))
      x = int(left*(frame_width/255))
      y = int(top*(frame_height/255))
      print("Bounding boxes: " + str([x,y,w,h]))
      cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
      cv2.putText(frame,class_names[i] + str(scores[i]),(x+w+10,y+h),0,0.5,(0,255,0),2)
    cv2.imshow('frame', frame)
    if(cv2.waitKey(25) & 0xFF == ord('q')):
        break

