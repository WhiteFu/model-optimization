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
# pylint: disable=g-import-not-at-top
"""MNIST training example for weight clustering algorithm using compression API."""
import os

from absl import app
from absl import flags
import tensorflow.compat.v2 as tf

from tensorflow_model_optimization.python.core.clustering.keras import cluster_config
from tensorflow_model_optimization.python.core.common.keras.compression.algorithms import weight_clustering
from tensorflow_model_optimization.python.examples.compression.keras import common

tf.enable_v2_behavior()


FLAGS = flags.FLAGS

flags.DEFINE_boolean('trim_dataset', False,
                     'turning on the debug mode, which evaluates the trained '
                     'models with the much smaller dataset.')
flags.DEFINE_integer('original_epochs', 1,
                     'Training epochs for original model.')
flags.DEFINE_integer('finetune_epochs', 1,
                     'Training epochs for compression api training model.')
flags.DEFINE_string('original_model_path', '/tmp/weight_clustering_original',
                    'Saved model path for original model.')
flags.DEFINE_string('original_tflite_path',
                    '/tmp/weight_clustering_original.tflite',
                    'Converted tflite file path for compressed model.')
flags.DEFINE_string('compressed_model_path',
                    '/tmp/weight_clustering_compressed',
                    'Saved model path for compressed model.')
flags.DEFINE_string('compressed_tflite_path',
                    '/tmp/weight_clustering_compressed.tflite',
                    'Converted tflite file path for compressed model.')


def weight_clustering_e2e():
  """WeightClusteringCompression e2e example."""
  ds = common.get_dataset(FLAGS.trim_dataset)
  original_model = common.training_original_model(
      ds, epochs=FLAGS.original_epochs)

  original_model.save(FLAGS.original_model_path)

  params = weight_clustering.WeightClusteringParams(
      number_of_clusters=8,
      cluster_centroids_init=\
      cluster_config.CentroidInitialization.DENSITY_BASED)
  compressed_model = weight_clustering.optimize(original_model, params)

  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  compressed_model.compile(optimizer='adam',
                           loss=loss_fn,
                           metrics=['accuracy'])

  print('Evaluate training model before training.')
  compressed_model.evaluate(ds.x_test, ds.y_test, verbose=2)

  compressed_model.fit(ds.x_train, ds.y_train, epochs=FLAGS.finetune_epochs)

  print('Evaluate training model after training.')
  compressed_model.evaluate(ds.x_test, ds.y_test, verbose=2)

  compressed_model.save(FLAGS.compressed_model_path)

  common.convert_to_tflite(FLAGS.original_model_path,
                           FLAGS.original_tflite_path)
  original_tflite_path_gz = '{}.gz'.format(FLAGS.original_tflite_path)
  common.compress_gzip(FLAGS.original_tflite_path,
                       original_tflite_path_gz)

  print('Evaluate original tflite inference')
  common.tflite_inference(ds, FLAGS.compressed_tflite_path)
  common.convert_to_tflite(FLAGS.compressed_model_path,
                           FLAGS.compressed_tflite_path)
  compressed_tflite_path_gz = '{}.gz'.format(FLAGS.compressed_tflite_path)
  common.compress_gzip(FLAGS.compressed_tflite_path,
                       compressed_tflite_path_gz)

  print('Evaluate compressed tflite inference')
  common.tflite_inference(ds, FLAGS.compressed_tflite_path)

  print('Original model saved model file size : {} bytes'.format(
      common.get_directory_size_in_bytes(FLAGS.original_model_path)))

  print('Original model TFLite file size : {} bytes (gzip : {} bytes)'.format(
      os.path.getsize(FLAGS.original_tflite_path),
      os.path.getsize(original_tflite_path_gz)))

  print('Compressed model saved model file size : {} bytes'.format(
      common.get_directory_size_in_bytes(FLAGS.compressed_model_path)))

  print('Compressed model TFLite file size : {} bytes (gzip : {} bytes)'.format(
      os.path.getsize(FLAGS.compressed_tflite_path),
      os.path.getsize(compressed_tflite_path_gz)))


def main(argv):
  del argv
  weight_clustering_e2e()


if __name__ == '__main__':
  app.run(main)
