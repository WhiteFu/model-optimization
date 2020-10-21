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
"""Tests for weight clustering algorithm."""

import os
import tempfile

import tensorflow as tf

from tensorflow_model_optimization.python.core.clustering.keras import cluster_config
from tensorflow_model_optimization.python.core.common.keras.compression.algorithms import weight_clustering


def _build_model():
  i = tf.keras.layers.Input(shape=(28, 28), name='input')
  x = tf.keras.layers.Reshape((28, 28, 1))(i)
  x = tf.keras.layers.Conv2D(
      20, 5, activation='relu', padding='valid', name='conv1')(
          x)
  x = tf.keras.layers.MaxPool2D(2, 2)(x)
  x = tf.keras.layers.Conv2D(
      50, 5, activation='relu', padding='valid', name='conv2')(
          x)
  x = tf.keras.layers.MaxPool2D(2, 2)(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(500, activation='relu', name='fc1')(x)
  output = tf.keras.layers.Dense(10, name='fc2')(x)

  model = tf.keras.Model(inputs=[i], outputs=[output])
  return model


def _get_dataset():
  mnist = tf.keras.datasets.mnist
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0
  # Use subset of 60000 examples to keep unit test speed fast.
  x_train = x_train[:1000]
  y_train = y_train[:1000]

  return (x_train, y_train), (x_test, y_test)


def _train_model(model):
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
  (x_train, y_train), _ = _get_dataset()
  model.fit(x_train, y_train, epochs=1)


def _save_as_saved_model(model):
  saved_model_dir = tempfile.mkdtemp()
  model.save(saved_model_dir)
  return saved_model_dir


def _get_directory_size_in_bytes(directory):
  total = 0
  try:
    for entry in os.scandir(directory):
      if entry.is_file():
        # if it's a file, use stat() function
        total += entry.stat().st_size
      elif entry.is_dir():
        # if it's a directory, recursively call this function
        total += _get_directory_size_in_bytes(entry.path)
  except NotADirectoryError:
    # if `directory` isn't a directory, get the file size then
    return os.path.getsize(directory)
  except PermissionError:
    # if for whatever reason we can't open the folder, return 0
    return 0
  return total


class FunctionalTest(tf.test.TestCase):

  def testWeightClustering_TrainingE2E(self):
    model = _build_model()
    _train_model(model)
    original_saved_model_dir = _save_as_saved_model(model)

    params = weight_clustering.WeightClusteringParams(
        number_of_clusters=8,
        cluster_centroids_init=\
        cluster_config.CentroidInitialization.DENSITY_BASED)
    compressed_model = weight_clustering.optimize(model, params)

    _train_model(compressed_model)

    saved_model_dir = _save_as_saved_model(compressed_model)

    _, (x_test, y_test) = _get_dataset()

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    compressed_model.compile(
        optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    results = compressed_model.evaluate(x_test, y_test)

    self.assertGreater(results[1], 0.85)  # 0.8708

    original_size = _get_directory_size_in_bytes(original_saved_model_dir)
    compressed_size = _get_directory_size_in_bytes(saved_model_dir)

    # TODO(tfmot): remove hardcoded sizes prior to checkin and only assert
    # that size is > Nx smaller, since can break if SavedModel serialization is
    # optimized.

    self.assertGreater(original_size, 5200000)  # 5323136
    self.assertLess(compressed_size, 3900000)  # 3733619


if __name__ == '__main__':
  tf.test.main()
