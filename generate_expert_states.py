from a2c_ppo_acktr.envs import make_vec_envs
import argparse
import torch
import os

def gen_data(args):
    num_envs = 10
    env = make_vec_envs(
        "PongNoFrameskip-v4",
        1000,
        num_envs,
        None,
        None,
        device='cuda',
        allow_early_resets=False)

    ppo, ob_rms = torch.load("models/PongNoFrameskip-v4.pt")
    recurrent_hidden_states = torch.zeros(1, ppo.recurrent_hidden_state_size).cuda()
    masks = torch.zeros(1, num_envs).cuda()
    rewards = torch.zeros(num_envs)
    obs = env.reset()
    steps = 0
    obs_train = []
    action_train = []
    total_rewards = 0
    num_trajectory = 0
    while steps < (args.num_steps//num_envs):
        steps += 1
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = ppo.act(
                obs, recurrent_hidden_states, masks, True)
        obs_train.append(obs.cpu())
        action_train.append(action.cpu())
        obs, reward, done, _ = env.step(action)
        rewards += reward.cpu().data.view(-1)
        for i, done_ in enumerate(done):
            if done_:
                total_rewards += rewards[i].item()
                num_trajectory += 1
                rewards[i] = 0
        masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
        
    env.close()
    obs_train = torch.cat(obs_train)
    action_train = torch.cat(action_train).view(-1)
    states = {
        "obs": obs_train,
        "action": action_train
    }
    print("Mean reward: {}".format(total_rewards/num_trajectory))
    save_file = os.path.join(args.save_path, "expert{}".format(args.num_steps))
    with open(save_file, mode="wb+") as f:
        torch.save(states, f)
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--save_path', type=str, default = "expert_states")
    gen_data(parser.parse_args())
    
if __name__ == "__main__":
    main()