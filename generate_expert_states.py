import a2c_ppo_acktr
from a2c_ppo_acktr.envs import make_vec_envs
import torch

env = make_vec_envs(
    "PongNoFrameskip-v4",
    1000,
    10,
    None,
    None,
    device='cuda',
    allow_early_resets=False)

ppo, ob_rms = torch.load("models/PongNoFrameskip-v4.pt")
recurrent_hidden_states = torch.zeros(1, ppo.recurrent_hidden_state_size).cuda()
masks = torch.zeros(1, 1).cuda()

obs = env.reset()
count = 0
obs_train = []
action_train = []
while True:
    count += 1
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = ppo.act(
            obs, recurrent_hidden_states, masks, True)
    obs_train.append(obs.data)
    action_train.append(action.data)
    # Obser reward and next obs
    obs, reward, done, _ = env.step(action)
    masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
    if count==5000:
        break

obs_train = torch.cat(obs_train)
action_train = torch.cat(action_train).view(-1)
states = {
    "obs": obs_train,
    "action": action_train
}
with open("expert_trajectory/expert50000", mode="wb+") as f:
    torch.save(states, f)