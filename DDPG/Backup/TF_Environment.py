
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

from tf_agents.networks import network

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.python.framework.indexed_slices import tensor_spec


from tf_agents.policies import random_py_policy, scripted_py_policy, random_tf_policy
import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

class Base(object):

  @abc.abstractmethod
  def __init__(self, time_step_spec, action_spec, policy_state_spec=()):
    self._time_step_spec = time_step_spec
    self._action_spec = action_spec
    self._policy_state_spec = policy_state_spec

  @abc.abstractmethod
  def reset(self, policy_state=()):
    # return initial_policy_state.
    pass

  @abc.abstractmethod
  def action(self, time_step, policy_state=()):
    # return a PolicyStep(action, state, info) named tuple.
    pass

  @abc.abstractmethod
  def distribution(self, time_step, policy_state=()):
    # Not implemented in python, only for TF policies.
    pass

  @abc.abstractmethod
  def update(self, policy):
    # update self to be similar to the input `policy`.
    pass

  @property
  def time_step_spec(self):
    return self._time_step_spec

  @property
  def action_spec(self):
    return self._action_spec

  @property
  def policy_state_spec(self):
    return self._policy_state_spec

class ActionNet(network.Network):

  def __init__(self, input_tensor_spec, output_tensor_spec, action_spec=None):
    super(ActionNet, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
        name='ActionNet')
    self._output_tensor_spec = output_tensor_spec
    self._sub_layers = [
        tf.keras.layers.Dense(
            action_spec.shape.num_elements(), activation=tf.nn.tanh),
    ]

  def call(self, observations, step_type, network_state):
    del step_type

    output = tf.cast(observations, dtype=tf.float32)
    for layer in self._sub_layers:
      output = layer(output)
    actions = tf.reshape(output, [-1] + self._output_tensor_spec.shape.as_list())

    # Scale and shift actions to the correct range if necessary.
    return actions, network_state