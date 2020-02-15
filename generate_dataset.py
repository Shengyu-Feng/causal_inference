import numpy as np
import matplotlib.pyplot as plt
from baselines import logger
from baselines.ppo2 import ppo2
from baselines.common.vec_env import VecFrameStack
from baselines.common.cmd_util import make_vec_env


def collect_states(model, env, limit):
    count = 0
    obs = env.reset()
    states = []
    state = model.initial_state if hasattr(model, 'initial_state') else None
    dones = np.zeros((1,))

    while True:
        if state is not None:
            actions, _, state, _ = model.step(obs,S=state, M=dones)
        else:
            actions, _, _, _ = model.step(obs)
        obs, rew, done, _ = env.step(actions)
        states.append(obs)
        count+=1
        if count >= limit:
            break;
            
    states = np.vstack(states)
    return states

def collect_demonstration(model,  env, limit):
    count = 0
    obs = env.reset()
    obs_list = []
    action_list = []
    state = model.initial_state if hasattr(model, 'initial_state') else None
    dones = np.zeros((1,))
    episode_num = 0
    rewards = 0
    while True:
        if state is not None:
            actions, _, state, _ = model.step(obs,S=state, M=dones)
        else:
            actions, _, _, _ = model.step(obs)
        obs_list.append(obs)
        action_list.append(actions)
        obs, rew, done, _ = env.step(actions)
        rewards += sum(rew)
        episode_num += sum(done)
        count+=1
        if count >= limit:
            break;
    print("Mean reward: {} for {} episodes".format(rewards/episode_num,episode_num))
            
    return np.vstack(obs_list), np.vstack(action_list).reshape(-1)
    
    
if __name__ == '__main__':
    seed = None
    frame_stack_size = 4
    # total <= num_envs*(expert_limit+random_limit) due to noisy states
    num_envs = 10
    limit = 10000
    expert_limit = 2000
    random_limit = 4000
    env = make_vec_env("PongNoFrameskip-v4", "atari", num_envs, seed, gamestate = None, reward_scale=1.0)
    env = VecFrameStack(env, frame_stack_size)

    model = ppo2.learn(
            env=env,
            seed=seed,
            total_timesteps=0,
            network = "cnn",
        )
    data = "demo"
    if data == "frame":
        states = collect_states(model, env, random_limit)
        model.load("./models/pong_20M_ppo2")
        states = np.vstack([states, collect_states(model, env, expert_limit)])
        env.close()
        states = np.moveaxis(states, -1, 1) # move axis if in pytorch, keep it if in tensorflow   
        np.random.shuffle(states)
        print("Generate {} stacked frames".format(len(states)))
        np.save("ministates.npy", states)
    else:
        model.load("./models/pong_20M_ppo2")
        obs, action = collect_demonstration(model, env, limit)
        obs = np.moveaxis(obs, -1, 1)
        states = {"obs": obs, "action": action}
        np.save("expert_states/expert100000.npy", states)

