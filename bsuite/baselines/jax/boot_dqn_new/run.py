# python3
# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Run a Dqn agent instance (using JAX) on a bsuite experiment."""

from absl import app
from absl import flags

import os
import bsuite
from bsuite import sweep

from bsuite.baselines import experiment
from agent import BootstrappedDqn
from bsuite.baselines.utils import pool

import haiku as hk
from jax import lax
from jax.config import config
import jax.numpy as jnp
import optax

# Internal imports.

flags.DEFINE_integer('num_ensemble', 1, 'Size of ensemble.')

# Experiment flags.
flags.DEFINE_string(
    'bsuite_id', 'catch/0', 'BSuite identifier. '
    'This global flag can be used to control which environment is loaded.')
flags.DEFINE_string('save_path', '/tmp/bsuite', 'where to save bsuite results')
flags.DEFINE_enum('logging_mode', 'csv', ['csv', 'sqlite', 'terminal'],
                  'which form of logging to use for bsuite results')
flags.DEFINE_boolean('overwrite', False, 'overwrite csv logging if found')
flags.DEFINE_integer('num_episodes', None, 'Number of episodes to run for.')
flags.DEFINE_boolean('verbose', True, 'whether to log to std output')
flags.DEFINE_float('penalty_weight', 0.01, 'the weight given to the gradient penalty')
flags.DEFINE_boolean('disable_jit', False, 'whether jit is disabled for debugging')
flags.DEFINE_integer('num_processes', 1, 'number of parellel processes')
flags.DEFINE_boolean('parellel', False, 'using concurrent.futures to enable parellelization')

FLAGS = flags.FLAGS

# Define the lite experiments
lite_experiments = {
    'DEEP_SEA_LITE': ['deep_sea/0', 'deep_sea/5', 'deep_sea/10', 'deep_sea/15'],
    'CARTPOLE_SWINGUP_LITE': ['cartpole_swingup/0', 'cartpole_swingup/5', 'cartpole_swingup/10', 'cartpole_swingup/15'],
}


def run(bsuite_id: str) -> str:
  """Runs a DQN agent on a given bsuite environment, logging to CSV."""

  env = bsuite.load_and_record(
      bsuite_id=bsuite_id,
      save_path=FLAGS.save_path,
      logging_mode=FLAGS.logging_mode,
      overwrite=FLAGS.overwrite,
  )
  action_spec = env.action_spec()

  # Define network.
  prior_scale = 5.
  hidden_sizes = [50, 50]
  def network(inputs: jnp.ndarray) -> jnp.ndarray:
    """Simple Q-network with randomized prior function."""
    net = hk.nets.MLP([*hidden_sizes, action_spec.num_values])
    prior_net = hk.nets.MLP([*hidden_sizes, action_spec.num_values])
    x = hk.Flatten()(inputs)
    return net(x) + prior_scale * lax.stop_gradient(prior_net(x))

  optimizer = optax.adam(learning_rate=1e-3)

  agent = BootstrappedDqn(
      obs_spec=env.observation_spec(),
      action_spec=action_spec,
      network=network,
      optimizer=optimizer,
      num_ensemble=FLAGS.num_ensemble,
      batch_size=128,
      discount=.99,
      replay_capacity=10000,
      min_replay_size=128,
      sgd_period=1,
      target_update_period=4,
      mask_prob=1.0,
      noise_scale=0.,
      penalty_weight=FLAGS.penalty_weight,
  )

  num_episodes = FLAGS.num_episodes or getattr(env, 'bsuite_num_episodes')
  experiment.run(
      agent=agent,
      environment=env,
      num_episodes=num_episodes,
      verbose=FLAGS.verbose)

  return bsuite_id


def main(_):
  # Parses whether to run a single bsuite_id, or multiprocess sweep.
  bsuite_id = FLAGS.bsuite_id

  # Set jax GPU memory limit
  os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.40'

  if FLAGS.disable_jit:
    config.update('jax_disable_jit', True)

  if bsuite_id in sweep.SWEEP:
    print(f'Running single experiment: bsuite_id={bsuite_id}.')
    run(bsuite_id)

  elif hasattr(sweep, bsuite_id) or bsuite_id in lite_experiments.keys():
    if hasattr(sweep, bsuite_id):
      bsuite_sweep = getattr(sweep, bsuite_id)
    else:
      bsuite_sweep = lite_experiments[bsuite_id]
      
    # FLAGS.verbose = False
    if FLAGS.parellel:
      print(f'Running sweep over bsuite_id in sweep.{bsuite_sweep}')
      pool.map_mpi(run, bsuite_sweep, num_processes=FLAGS.num_processes)
    else:
      for bsuite_id in bsuite_sweep:
        print(f'Running single experiment: bsuite_id={bsuite_id}.')
        run(bsuite_id)

  else:
    raise ValueError(f'Invalid flag: bsuite_id={bsuite_id}.')


if __name__ == '__main__':
  app.run(main)
