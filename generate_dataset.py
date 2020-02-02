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
    states_sum = states.sum(axis=(1,2,3))
    states = states[np.where(states_sum!=749607)]
    return states

if __name__ == '__main__':
    seed = None
    frame_stack_size = 4
    # total <= num_envs*(expert_limit+random_limit) due to noisy states
    num_envs = 50
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

    states = collect_states(model, env, random_limit)
    model.load("./models/pong_20M_ppo2")
    states = np.vstack([states, collect_states(model, env, expert_limit)])
    env.close()
    states = np.moveaxis(states, -1, 1) # move axis if in pytorch, keep it if in tensorflow   
    np.random.shuffle(states)
    print("Generate {} stacked frames".format(len(states)))
    np.save("ministates.npy", states)

