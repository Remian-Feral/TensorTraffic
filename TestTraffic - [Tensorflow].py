# region Import Dependencies
import os, sys, optparse

from Cython import returns
from gym.spaces import Discrete

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Removing CudaNN Warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tf_agents.agents.ddpg import actor_network
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.ddpg import ddpg_agent

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import policy_step
from tf_agents.utils import common
import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register
import numpy as np

if 'SUMO_HOME' in os.environ:  # Checks for Environment Variable named 'SUMO_HOME'
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci  # TraCI functionality
from sumolib import checkBinary  # checkBinary help locate the Sumo binaries
import time

# endregion
# traci.connect()
traci.start(['sumo-gui', '-c', 'Neural Traffic.sumocfg', "--start", "--quit-on-end"])


class NeuralTraffic(gym.Env):
    LaneStates = 4
    Lanes = ['North', 'South', 'East', 'West']

    Northbound = 0
    Southbound = 1
    F_Ponce_Road = 2
    Pineda_Road = 3
    AVG_HALTING_TIME = 4

    North = (traci.lanearea.getLastStepVehicleNumber("Nation_Highway_North_Outer") +
             traci.lanearea.getLastStepVehicleNumber("Nation_Highway_North_Inner"))

    South = (traci.lanearea.getLastStepVehicleNumber("Nation_Highway_South_Outer") +
             traci.lanearea.getLastStepVehicleNumber("Nation_Highway_South_Inner"))
    East = traci.lanearea.getLastStepVehicleNumber("San_Pedro_Libis")
    West = traci.lanearea.getLastStepVehicleNumber("Pineda")

    Step = 0
    Done = False
    SUM_HALTING_TIME = 0

    HALTING_TIME = 0
    Prev_HALTING_TIME = 0
    Reward = None

    def __init__(self):  # Initialize the Action and Observation Space
        self.verbose = False
        self.viewer = None
        self.action_space = spaces.Box(low=np.int32(np.array([0, 0])), high=np.float32(np.array([1, 120])))
        self.observation_space = spaces.Box(low=np.float32(np.array([0, 0, 0, 0, 0])),
                                            high=np.float32(np.array([207, 211, 79, 87, np.inf])))
        self.state = [self.North, self.South, self.East, self.West,  # Declaration of variables
                      self.HALTING_TIME]
        self.Prev_HALTING_TIME
        self.seed()

    def seed(self, seed=None):  # Seed No. for reproducibility
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        Reward = 0
        info = []
        if self.Step < 3600:
            # region The traffic part
            # ----------------------------------Traffic Control -----------------------------------------------------
            North = (traci.lanearea.getLastStepVehicleNumber("Nation_Highway_North_Outer") +
                     traci.lanearea.getLastStepVehicleNumber("Nation_Highway_North_Inner"))

            South = (traci.lanearea.getLastStepVehicleNumber("Nation_Highway_South_Outer") +
                     traci.lanearea.getLastStepVehicleNumber("Nation_Highway_South_Inner"))
            East = traci.lanearea.getLastStepVehicleNumber("San_Pedro_Libis")
            West = traci.lanearea.getLastStepVehicleNumber("Pineda")

            SUM_HALTING_TIME = (traci.lanearea.getLastStepHaltingNumber("Nation_Highway_North_Outer") +
                                traci.lanearea.getLastStepHaltingNumber("Nation_Highway_North_Inner")) + \
                               (traci.lanearea.getLastStepHaltingNumber("Nation_Highway_South_Outer") +
                                traci.lanearea.getLastStepHaltingNumber("Nation_Highway_South_Inner")) + \
                               traci.lanearea.getLastStepVehicleNumber("San_Pedro_Libis") + \
                               traci.lanearea.getLastStepVehicleNumber("Pineda")

            HALTING_TIME = SUM_HALTING_TIME / 4
            Prev_HALTING_TIME = None

            self.state = [North / 584, South / 584, East / 584, West / 584,
                          HALTING_TIME]

            Step = 0

            Phase, Duration = action

            Phase = int(round(Phase))
            Duration = int(Duration)
            print('Phase value: {:4f} Duration Value: {}'.format(Phase, Duration))

            if Phase == 0:
                for i in range(0, Duration):
                    traci.trafficlight.setPhase('320811091', 0)
                    traci.trafficlight.setPhase('469173108', 0)
                    traci.simulationStep()
                    time.sleep(0.05)
                    Step += 1
                for i in range(0, 20):
                    traci.trafficlight.setPhase('320811091', 1)
                    traci.trafficlight.setPhase('469173108', 1)
                    traci.simulationStep()
                    time.sleep(0.05)
                    Step += 1

            elif Phase == 1:
                for i in range(0, Duration):
                    traci.trafficlight.setPhase('320811091', 2)
                    traci.trafficlight.setPhase('469173108', 2)
                    traci.simulationStep()
                    time.sleep(0.05)
                    Step += 1
                for i in range(0, 20):
                    traci.trafficlight.setPhase('320811091', 3)
                    traci.trafficlight.setPhase('469173108', 3)
                    traci.simulationStep()
                    time.sleep(0.05)
                    Step += 1
            self.Step += Step

            print(HALTING_TIME)
            if self.Prev_HALTING_TIME == 0 and HALTING_TIME <= 5000:
                self.Prev_HALTING_TIME = HALTING_TIME
                Reward = 0
                print('The Reward:{} PrevHaltingTime:{} HaltingTime:{}'.format(Reward, self.Prev_HALTING_TIME,
                                                                               HALTING_TIME))

            elif self.Prev_HALTING_TIME != 0 and HALTING_TIME <= 5000:
                if self.Prev_HALTING_TIME == HALTING_TIME and self.isEmpty:
                    Reward -= HALTING_TIME
                else:
                    Reward += self.Prev_HALTING_TIME - HALTING_TIME
                print('The Reward:{} PrevHaltingTime:{} HaltingTime:{}'.format(Reward, self.Prev_HALTING_TIME,
                                                                               HALTING_TIME))
                self.Prev_HALTING_TIME = HALTING_TIME

            self.state = [North / 584, South / 584, East / 584, West / 584,
                          HALTING_TIME]
            Done = False
            # ---------------------------------------------------------------------------------------------------------
            # endregion
        else:
            Done = True
        return self.state, Reward, Done, info

    def isEmpty(self):  # Checks for Empty Intersections

        North = (traci.lanearea.getLastStepVehicleNumber("Nation_Highway_North_Outer") +
                 traci.lanearea.getLastStepVehicleNumber("Nation_Highway_North_Inner"))

        South = (traci.lanearea.getLastStepVehicleNumber("Nation_Highway_South_Outer") +
                 traci.lanearea.getLastStepVehicleNumber("Nation_Highway_South_Inner"))
        East = traci.lanearea.getLastStepVehicleNumber("San_Pedro_Libis")
        West = traci.lanearea.getLastStepVehicleNumber("Pineda")

        NEWS = North + South + East + West

        if NEWS:
            return False
        else:
            return True

    def render(self):
        pass
        # TODO this is where the excelsheet comes into play

    def reset(self):
        traci.close()
        traci.start(['sumo-gui', '-c', 'Neural Traffic.sumocfg', "--start", "--quit-on-end"])
        LaneStates = 4
        Lanes = ['North', 'South', 'East', 'West']

        Northbound = 0
        Southbound = 1
        F_Ponce_Road = 2
        Pineda_Road = 3
        AVG_HALTING_TIME = 4

        North = (traci.lanearea.getLastStepVehicleNumber("Nation_Highway_North_Outer") +
                 traci.lanearea.getLastStepVehicleNumber("Nation_Highway_North_Inner"))

        South = (traci.lanearea.getLastStepVehicleNumber("Nation_Highway_South_Outer") +
                 traci.lanearea.getLastStepVehicleNumber("Nation_Highway_South_Inner"))
        East = traci.lanearea.getLastStepVehicleNumber("San_Pedro_Libis")
        West = traci.lanearea.getLastStepVehicleNumber("Pineda")


register(
    id='NeuralTraffic-v1',
    entry_point=f'{__name__}:NeuralTraffic'
)

env = gym.make('NeuralTraffic-v1')

episodes = 1

for episode in range(0, episodes):
    state = env.reset()
    done = False
    Score = 0

    while not done:
        action = env.action_space.sample()
        N_state, reward, done, info = env.step(action)
        Score += reward
        print('Current Score:{:15f}'.format(Score))
    print('Episode:{} Score:{:15f}'.format(episode, Score))
env.close()

# Num_iterations = 50000
#
# inital_collect_steps = 1000
#
# collect_steps_per_iteration = 50
#
# replay_buffer_max_length = 100000
#
# batch_size = 64
#
# log_interval = 2500
# num_eval_episodes = 2500
#
# eval_interval = 5000
#
# train_py = suite_gym.load('NeuralTraffic-v1')
# eval_py = suite_gym.load('NeuralTraffic-v1')
# train_env = tf_py_environment.TFPyEnvironment(train_py)
# eval_env = tf_py_environment.TFPyEnvironment(eval_py)
#
# actor_fc_layers = (400, 300)
# critic_obs_fc_layers = (400,)
# critic_action_fc_layers = None
# critic_joint_fc_layers = (300,)
# ou_stddev = 0.2
# ou_damping = 0.15
# target_update_tau = 0.05
# target_update_period = 5
# dqda_clipping = None
# td_errors_loss_fn = tf.compat.v1.losses.huber_loss
# gamma = 0.995
# reward_scale_factor = 1.0
# gradient_clipping = None
#
# actor_learning_rate = 1e-4
# critic_learning_rate = 1e-3
#
# debug_summaries = False
# summarize_grads_and_vars = False
# global_step = tf.compat.v1.train.get_or_create_global_step()
#
# actor_net = actor_network.ActorNetwork(
#     train_env.time_step_spec().observation,
#     train_env.action_spec(),
#     fc_layer_params=actor_fc_layers,
# )
#
# critic_net_input_specs = (train_env.time_step_spec().observation,
#                           train_env.action_spec())
#
# critic_net = critic_network.CriticNetwork(
#     critic_net_input_specs,
#     observation_fc_layer_params=critic_obs_fc_layers,
#     joint_fc_layer_params=critic_joint_fc_layers,
# )
#
# tf_agent = ddpg_agent.DdpgAgent(
#     train_env.time_step_spec(),
#     train_env.action_spec(),
#     actor_network=actor_net,
#     critic_network=critic_net,
#     actor_optimizer=tf.compat.v1.train.AdamOptimizer(
#         learning_rate=actor_learning_rate),
#     critic_optimizer=tf.compat.v1.train.AdamOptimizer(
#         learning_rate=critic_learning_rate),
#     ou_stddev=ou_stddev,
#     ou_damping=ou_damping,
#     target_update_tau=target_update_tau,
#     target_update_period=target_update_period,
#     dqda_clipping=dqda_clipping,
#     td_errors_loss_fn=td_errors_loss_fn,
#     gamma=gamma,
#     reward_scale_factor=reward_scale_factor,
#     gradient_clipping=gradient_clipping,
#     debug_summaries=debug_summaries,
#     summarize_grads_and_vars=summarize_grads_and_vars,
#     train_step_counter=global_step)
# tf_agent.initialize()
#
#
# def compute_avg_return(environment, policy, num_episodes=10):
#     total_return = 0.0
#     for _ in range(num_episodes):
#
#         time_step = environment.reset()
#         episode_return = 0.0
#
#         while not time_step.is_last():
#             action_step = policy.action(time_step)
#             time_step = environment.step(action_step.action)
#             episode_return += time_step.reward
#         total_return += episode_return
#
#     avg_return = total_return / num_episodes
#     return avg_return.numpy()[0]
#
#
# def collect_step(environment, policy, buffer):
#     time_step = environment.current_time_step()
#     action_step = policy.action(time_step)
#     next_time_step = environment.step(action_step)
#     traj = trajectory.from_transition(time_step, action_step, next_time_step)
#
#     buffer.add_batch(traj)
#
#
# def collect_data(env, policy, buffer, steps):
#     for _ in range(steps):
#         collect_step(env, policy, buffer)
#
#
# random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())
#
# replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
#     data_spec=tf_agent.collect_data_spec,
#     batch_size=train_env.batch_size,
#     max_length=replay_buffer_max_length
# )
#
# collect_data(train_env, random_policy, replay_buffer, steps=100)
#
# dataset = replay_buffer.as_dataset(
#     num_parallel_calls=3,
#     sample_batch_size=batch_size,
#     num_steps=2).prefetch(3)
#
# iterator = iter(dataset)
#
# tf_agent.train = common.function(tf_agent.train)
#
# tf_agent.train_step_counter.assign(0)
#
# avg_return = compute_avg_return(eval_env, tf_agent.collect_policy, num_eval_episodes)
#
# for _ in range(Num_iterations):
#
#     for _ in range(collect_steps_per_iteration):
#         collect_step(train_env, tf.agent.collect_policy, replay_buffer)
#
#     experience, unused_info = next(iterator)
#
#     train_loss = tf.agent.train(experience).loss
#
#     step = tf_agent.train_step_counter.numpy()
#
#     if step % log_interval == 0:
#         print('step = {0}: loss = {1}'.format(step, train_loss))
#
#     if step % eval_interval == 0:
#         avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
#         print('step = {0}: Average Return = {1}'.format(step, avg_return))
#         returns.append(avg_return)
# step = 0  # variable saves the number of current simulation steps
# while step < 3600:
#     traci.simulationStep()
#     step += 1
#     time.sleep(0.05)
#
#     if (step % 50) == 0:
#         North = (traci.lanearea.getLastStepVehicleNumber("Nation_Highway_North_Outer") +
#                  traci.lanearea.getLastStepVehicleNumber("Nation_Highway_North_Inner"))
#         South = (traci.lanearea.getLastStepVehicleNumber("Nation_Highway_South_Outer") +
#                  traci.lanearea.getLastStepVehicleNumber("Nation_Highway_South_Inner"))
#         East = traci.lanearea.getLastStepVehicleNumber("San_Pedro_Libis")
#         West = traci.lanearea.getLastStepVehicleNumber("Pineda")
#
#         NeuralTrf = "Neural Traffic Demands"
#         print(NeuralTrf.center(60, '-'))
#
#         print(f"North National Highway: {North} || South National Highway: {South}")
#         print(f"F. Ponce de Leon Road: {East} || Pineda Road: {West}")
#
# traci.close()
