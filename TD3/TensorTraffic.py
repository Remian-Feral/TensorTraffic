import gym
import numpy as np

import xlsxwriter

import Neural_Traffic_Env

import stable_baselines3
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList, StopTrainingOnMaxEpisodes, EveryNTimesteps

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor, ResultsWriter
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

env = gym.make('NeuralTraffic-v1')
env = Monitor(env, filename="Monitor")
# env = DummyVecEnv([lambda : env])

# model = TD3.load("td3")


eval_callback = EvalCallback(env, best_model_save_path='./logs/best_model', log_path='./logs/', eval_freq=7000)
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./saves/')
callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=1, verbose=1)
callback = CallbackList([callback_max_episodes, checkpoint_callback, eval_callback])

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1, learning_rate=0.5)
model.learn(total_timesteps=1e6, log_interval=1, callback=callback)
model.save("td3")

# env.results_writer()

WrkBk = xlsxwriter.Workbook(f"./Trajectories/Convergence.xlsx")
WrkSht = WrkBk.add_worksheet(f'Trajectory')
WrkSht.write('A1', 'Episode Times')
WrkSht.write('B1', 'Episode Lengths')
WrkSht.write('C1', 'Episode Rewards')
WrkSht.write('D1', 'Episodic Timesteps')
WrkSht.set_column(0, 12, width=20)

Episode_Times = env.get_episode_times()
Episode_Lengths = env.get_episode_lengths()
Episode_Rewards = env.get_episode_rewards()
Episodic_Timestep = env.get_total_steps()


for _ in range(len(Episode_Times)):
    x = _ + 2
    WrkSht.write('A' + str(x), Episode_Times[_])
    WrkSht.write('B' + str(x), Episode_Lengths[_])
    WrkSht.write('C' + str(x), Episode_Rewards[_])
WrkBk.close()
print(Episodic_Timestep)
env = model.get_env()


# for _ in range(10):
#     if _ == 0:
#         model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
#         model.learn(total_timesteps=1, log_interval=1)
#         model.save("ddpg.pth")
#         env = model.get_env()
#         del model
#     else:
#         model = DDPG.load("ddpg.pth")
#         traci.close()
#         model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
#         model.learn(total_timesteps=1, log_interval=1)
#         model.save("ddpg.pth")
#
# del model
# model = DDPG.load("ddpg.pth")
# traci.close()
# obs = env.reset()
# dones = False
# while not dones:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)

# model = td3('CnnPolicy', env=env, verbose=1)
# model.learn(total_timesteps=3600)

# score_history = []
#
# num_episodes = 10
# for episode in range(0, num_episodes):
#     done = False
#     Score = 0
#     observation = env.reset()
#     env.step()
#     while not done:
#         action = numpy.array(agent.choose_actions(observation)).shape(1,2)
#         observation_, reward, done, info = env.step(action)
#         agent.learn(observation, reward, observation_, done)
#         observation = observation_
#         Score += reward
#         print('Current Score:{:15f}'.format(Score))
#     print('Episode:{} Score:{:15f}'.format(episode, Score))

    # model = A2C('MlpPolicy', 'NeuralTraffic-v1', verbose=0)
    #
    # model.learn(total_timesteps=10000)
    #
    # model.save("TrafficModel-v1")

    # obs = env.reset()
    #
    # while True:
    #     action, _states = model.predict(observation=env.observation_space)
    #     print(action)
    #     obs, rewards, done, info = env.step(action)



# episodes = 1
#
# for episode in range(0, episodes):
#     state = env.reset()
#     done = False
#     Score = 0
#
#     while not done:
#         action = env.action_space.sample()
#         N_state, reward, done, info = env.step(action)
#         Score += reward
#         print('Current Score:{:15f}'.format(Score))
#     print('Episode:{} Score:{:15f}'.format(episode, Score))
# env.close()
