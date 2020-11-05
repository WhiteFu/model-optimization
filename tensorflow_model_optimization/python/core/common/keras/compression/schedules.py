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
"""Compression Schedule functions for tfmot compression."""
import abc
import tensorflow.compat.v2 as tf


class Scheduler(tf.Module, abc.ABC):
  """Abstract Scheduler."""

  def __init__(self, name=None):
    super(Scheduler, self).__init__(name=name)

  @abc.abstractmethod
  def __call__(self, iterations):
    pass


class PolynomialDecay(Scheduler):
  """Scheduling based on polynomial equation."""

  def __init__(self, start_value, end_value, decay_steps,
               begin_step=0, exponent=3, dtype=tf.int32, name=None):
    super(PolynomialDecay, self).__init__(name=name)
    self.start_value = start_value
    self.end_value = end_value
    self.begin_step = begin_step
    self.decay_steps = decay_steps
    self.end_step = self.begin_step + self.decay_steps
    self.exponent = exponent
    self.dtype = dtype

  def __call__(self, iterations):

    def _during_decay():
      local_steps = tf.cast(iterations - self.begin_step, dtype=tf.float32)
      decay_steps = tf.cast(self.decay_steps, dtype=tf.float32)
      decay_term = tf.pow((local_steps / decay_steps), self.exponent)
      total_delta = tf.cast(self.end_value - self.start_value, dtype=tf.float32)
      target = self.start_value + tf.cast(total_delta * decay_term,
                                          dtype=self.dtype)
      return tf.stop_gradient(target)

    def _after_begin_step():
      return tf.cond(tf.math.greater(iterations, self.end_step),
                     lambda: self.end_value,
                     _during_decay, name="end")

    return tf.cond(tf.math.less(iterations, self.begin_step),
                   lambda: self.start_value,
                   _after_begin_step, name="start")
