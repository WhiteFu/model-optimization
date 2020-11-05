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
"""Tests for compress wrappers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_model_optimization.python.core.common.keras.compression import schedules

keras = tf.keras
layers = keras.layers

tf.enable_v2_behavior()


class SimpleScheduler(schedules.Scheduler):

  def __call__(self, iterations):
    return 0.1 if iterations >= 1000 else 0.6


class SimpleSchedulerTest(tf.test.TestCase):

  def testSimpleScheduler(self):
    scheduler = SimpleScheduler()

    expected_output = [0.6, 0.6, 0.1, 0.1]
    output = []
    for i in [0, 100, 1000, 2000]:
      output.append(scheduler(i))
    self.assertAllEqual(output, expected_output)


class PolynomialDecaySchedulerTest(tf.test.TestCase):

  def testPolynomialDecayScheduler(self):
    init_value = 0.1
    final_value = 1.0
    begin_step = 10
    decaying_step = 10
    total_training_step = 30
    scheduler = schedules.PolynomialDecay(init_value, final_value,
                                          decaying_step, begin_step=begin_step,
                                          dtype=tf.float32)

    before_decaying = [init_value] * begin_step
    decaying = [0.1, 0.1009, 0.1072, 0.1243, 0.1576,
                0.2125, 0.2944, 0.4087, 0.5608, 0.7561]
    after_decaying = [final_value] * (
        total_training_step - decaying_step - begin_step)
    expected_output = before_decaying + decaying + after_decaying
    output = []
    for i in range(total_training_step):
      output.append(scheduler(i))
    self.assertAllClose(output, expected_output)


if __name__ == '__main__':
  tf.test.main()
