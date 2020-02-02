import numpy as np
import matplotlib.pyplot as plt
from baselines import logger
from baselines.ppo2 import ppo2
from baselines.common.vec_env import VecFrameStack
from baselines.common.cmd_util import make_vec_env

if __name__ == '__main__':
    seed = None
    frame_stack_size = 4
    num_envs = 50
    env = make_vec_env("PongNoFrameskip-v4", "atari", num_envs, seed, gamestate = None, reward_scale=1.0)
    env = VecFrameStack(env, frame_stack_size)

    model = ppo2.learn(
            env=env,
            seed=seed,
            total_timesteps=0,
            network = "cnn",
            load_path = "./models/pong_20M_ppo2",
        )

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
        if count >= 10:
            break;

    states = np.vstack(states)
    states = np.moveaxis(states, -1, 1)
    np.save("toy_states.npy", states)

