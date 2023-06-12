# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Tensorflow Example proto decoder for object detection.

A decoder to decode string tensors containing serialized tensorflow.Example
protos for object detection.
"""
import json
import os
import numpy as np
import tensorflow as tf
from official.vision.dataloaders import decoder
import common
import PerIL.common
import PerIL.meds
import cv2 as cv


def _generate_source_id(image_bytes):
  # Hashing using 22 bits since float32 has only 23 mantissa bits.
  return tf.strings.as_string(tf.strings.to_hash_bucket_fast(image_bytes, 2 ** 22 - 1))


class ViaDecoder(decoder.Decoder):
  def __init__(self, data_subset_dir, class_names, include_mask=False, preprocess=True):
    annotations_json_path = os.path.join(data_subset_dir, 'annotations.json')
    with open(annotations_json_path, 'r') as f:
        self._annotations = json.load(f)
    self._image_dir = os.path.split(annotations_json_path)[0]
    self._class_names = class_names
    self._include_mask = include_mask
    self._keys_to_features = {'image/key': tf.io.FixedLenFeature((), tf.string)}
    self._preprocess = preprocess
    if self._preprocess:
        self._preprocessor = common.Preprocessor()

  def decode(self, serialized_example):
    """Decode the serialized example.

    Args:
      serialized_example: a single serialized tf.Example string.

    Returns:
      decoded_tensors: a dictionary of tensors with the following fields:
        - source_id: a string scalar tensor.
        - image: a tensor of shape [None, None, 3].
        - height: an integer scalar tensor.
        - width: an integer scalar tensor.
        - groundtruth_classes: a int64 tensor of shape [None].
        - groundtruth_is_crowd: a bool tensor of shape [None].
        - groundtruth_area: a float32 tensor of shape [None].
        - groundtruth_boxes: a float32 tensor of shape [None, 4].
        - groundtruth_instance_masks: a float32 tensor of shape
            [None, None, None].
    """
    parsed_tensors = tf.io.parse_single_example(serialized=serialized_example, features=self._keys_to_features)
    image_key = parsed_tensors['image/key']
    image_annotations = self._annotations[image_key]
    image_filename = image_annotations['filename']
    image_path = os.path.join(self._image_dir, image_filename)
    source_id = _generate_source_id(image_path)
    image = PerIL.common.load_image(image_path)

    if self._preprocess:
        _, _, image, perspective_transform = self._preprocessor.preprocess(image)
    else:
        perspective_transform = None

    regions = image_annotations['regions']

    height, width = image.shape[0:2]
    classes, is_crowds, areas, boxes, masks = [], [], [], [], []

    for i, region in enumerate(regions):
        class_name = common.region_attributes_to_class_name(region['region_attributes'])
        class_id = self._class_names.index(class_name)
        classes.append(class_id)

        is_crowds.append(False)

        contour = common.shape_attributes_to_contour(region['shape_attributes'])
        if perspective_transform is not None:
            contour = common.transform_contour(contour, perspective_transform)

        l, t, w, h = cv.boundingRect(contour)
        r, b = l + w, t + h

        areas.append(w * h)

        t = float(t) / height
        l = float(l) / width
        b = float(b) / height
        r = float(r) / width

        boxes.append([t, l, b, r])

        if self._include_mask:
            mask = np.zeros(shape=(height, width), dtype=np.uint8)
            cv.drawContours(mask, [contour], contourIdx=-1, color=1, thickness=-1)
            masks.append(mask)

    decoded_tensors = {
        'source_id': source_id,
        'image': image,
        'height': height,
        'width': width,
        'groundtruth_classes': classes,
        'groundtruth_is_crowd': is_crowds,
        'groundtruth_area': areas,
        'groundtruth_boxes': boxes,
    }
    if self._include_mask:
      decoded_tensors['groundtruth_instance_masks'] = masks

    return decoded_tensors
