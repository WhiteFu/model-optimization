# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Common function for compression examples."""

import gzip
import os
import shutil

import attr
import numpy as np
import tensorflow.compat.v2 as tf

tf.enable_v2_behavior()


@attr.s(frozen=True)
class MNISTDataset:
  x_train = attr.ib()
  y_train = attr.ib()
  x_test = attr.ib()
  y_test = attr.ib()


def get_dataset(trim=True):
  """get MNIST dataset."""
  mnist = tf.keras.datasets.mnist
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0
  if trim:
    print(' [*] Trim the dataset...')
    x_train = x_train[:100]
    y_train = y_train[:100]
    x_test = x_test[:100]
    y_test = y_test[:100]
  return MNISTDataset(x_train, y_train, x_test, y_test)


def training_original_model(ds, epochs=5, train=True):
  """Train original vanilla model."""
  input_layer = tf.keras.layers.Input(shape=(28, 28), name='input')
  x = tf.keras.layers.Reshape((28, 28, 1))(input_layer)
  x = tf.keras.layers.Conv2D(
      20, 5, activation='relu', padding='valid', name='conv1')(x)
  x = tf.keras.layers.MaxPool2D(2, 2)(x)
  x = tf.keras.layers.Conv2D(
      50, 5, activation='relu', padding='valid', name='conv2')(x)
  x = tf.keras.layers.MaxPool2D(2, 2)(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(500, activation='relu', name='fc1')(x)
  output_layer = tf.keras.layers.Dense(10, name='fc2')(x)
  model = tf.keras.Model(inputs=[input_layer], outputs=[output_layer])

  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

  model.compile(optimizer='adam',
                loss=loss_fn,
                metrics=['accuracy'])

  if train:
    model.fit(ds.x_train, ds.y_train, epochs=epochs)
    print('Evaluate model')
    model.evaluate(ds.x_test, ds.y_test, verbose=2)
  return model


def convert_to_tflite(saved_model_path, tflite_path):
  """Save the saved file and convert that to tflite."""
  converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
  # converter.target_spec.supported_types = []
  converted = converter.convert()
  open(tflite_path, 'wb').write(converted)


def tflite_inference(ds, tflite_path, num_test=-1):
  """TFLite inference and get accuracy."""
  # Load TFLite model and allocate tensors.
  interpreter = tf.lite.Interpreter(model_path=tflite_path)
  interpreter.allocate_tensors()

  # Get input and output tensors.
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  if num_test == -1:
    num_test = len(ds.x_test)

  correct = 0
  incorrect = 0
  for i in range(num_test):
    interpreter.set_tensor(
        input_details[0]['index'], ds.x_test[i:i+1].astype(np.float32))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    if ds.y_test[i] == output_data.argmax():
      correct += 1
    else:
      incorrect += 1

  accuracy = correct / (correct + incorrect)
  print('tflite inference accuracy = {}'.format(accuracy))


def get_directory_size_in_bytes(directory):
  """Get directory size in bytes."""

  total = 0
  try:
    for entry in os.scandir(directory):
      if entry.is_file():
        # if it's a file, use stat() function
        total += entry.stat().st_size
      elif entry.is_dir():
        # if it's a directory, recursively call this function
        total += get_directory_size_in_bytes(entry.path)
  except NotADirectoryError:
    # if `directory` isn't a directory, get the file size then
    return os.path.getsize(directory)
  except PermissionError:
    # if for whatever reason we can't open the folder, return 0
    return 0
  return total


def compress_gzip(source, target):
  with open(source, 'rb') as f_in:
    with gzip.open(target, 'wb') as f_out:
      shutil.copyfileobj(f_in, f_out)

