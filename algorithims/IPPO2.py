import torch
import torch.nn.functional as F
import numpy as np
import utils.rl_utils
from tqdm import tqdm
import matplotlib.pyplot as plt
import utils.get_h
import utils.get_λ
import sys

from env.env_exploration import GridWorld

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        # Subtract max to ensure numerical stability
        logits = self.fc3(x)
        logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
        return F.softmax(logits, dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return self.fc3(x)


class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)

        # Check for NaNs in probabilities
        assert not torch.isnan(probs).any(), "NaN in action probabilities!"
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        h = torch.tensor(transition_dict['h'], dtype=torch.float).view(-1, 1).to(self.device)
        λ = torch.tensor(transition_dict['λ'], dtype=torch.float).view(-1, 1).to(self.device)
        rewards = rewards + gamma * λ * h  # 计算新奖励
        new_gamma = self.gamma * (1 - λ)  # 计算新折扣因子
        td_target = rewards + new_gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = utils.rl_utils.compute_advantage(new_gamma, self.lmbda, td_delta.cpu()).to(self.device)

        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()
        log_probs = torch.log(self.actor(states).gather(1, actions))
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
        actor_loss = torch.mean(-torch.min(surr1, surr2))
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()

        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)

        self.actor_optimizer.step()
        self.critic_optimizer.step()
def greedy_exploration(world):
    actions = []
    for i, state in enumerate(world.states):
        x, y = int(state[0]), int(state[1])  # 当前机器人的位置

        # 获取周围四个方向的状态
        candidates = {
            1: (x - 1, y),  # 上
            2: (x + 1, y),  # 下
            3: (x, y - 1),  # 左
            4: (x, y + 1)  # 右
        }

        # 筛选出合法的、且未被探索的格子
        valid_moves = []
        for action, (nx, ny) in candidates.items():
            if not world._is_out_of_bounds(nx, ny) and world.grid[nx, ny] == 0:  # 如果格子未被探索
                valid_moves.append((action, nx, ny))

        # 如果有未被探索的格子，选择其中之一
        if valid_moves:
            chosen_action, chosen_x, chosen_y = valid_moves[0]  # 选择第一个未探索的格子作为目标
        else:
            # 如果没有未被探索的格子或周围是障碍物，则随机选择一个动作
            chosen_action = np.random.randint(1, 5)

        actions.append(chosen_action)

    return actions
def calculate_h(rewards, gamma):
    T_r = len(rewards)
    h_values = []
    for t in range(T_r):
        h_t = sum(gamma ** (k - t) * rewards[k] for k in range(t, T_r))
        h_values.append(h_t)
    return h_values
actor_lr = 3e-4
critic_lr = 1e-3
num_episodes = 5000
hidden_dim = 64
gamma = 0.9
lmbda = 0.97
eps = 0.2
ncols = 15
nrows = 15
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

team_size = 3

env = GridWorld(nrows,ncols,num_robots=team_size)
state_dim = 4
action_dim = 5

agent_group = [PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, eps, gamma, device) for _ in range(3)]
win_list = []
step_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            transitions = [{'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [],'h':[],'λ':[]} for _ in range(3)]
            state = env.reset()
            done = False
            while not done and env.step_count<500:
                actions = []
                for i in range(3):
                    actions.append(agent_group[i].take_action(state[i][2:]))
                    # pos_i = env.agent_pos[i]

                next_state, reward, done= env.step(actions)
                # env.plot()
                for i in range(3):
                    transitions[i]['states'].append(state[i][2:])
                    transitions[i]['actions'].append(actions[i])
                    transitions[i]['next_states'].append(next_state[i][2:])
                    transitions[i]['rewards'].append(reward[i])
                    transitions[i]['dones'].append(done)

                state = next_state
                terminal =done
            print(env.step_count)
            print(env.explore_rate)
            for i in range(team_size):
                rewards = torch.tensor(transitions[i]['rewards'], dtype=torch.float).view(-1, 1)

                transitions[i]['h']= utils.get_h.calculate_h(rewards)
                h = torch.tensor(transitions[i]['h'], dtype=torch.float).view(-1, 1)
                for i in range(len(h)):

                    transitions[i]['λ'].append(utils.get_λ.lambda_constant(h[i]))

                new_r = reward[i] + gamma * lambda_value * h_prime  # 计算新奖励
                new_gamma = gamma * (1 - lambda_prime)  # 计算新折扣因子

                new_rewards.append(new_r)
                new_discounts.append(new_gamma)

                # 更新 transitions 结构
                transitions[i]['rewards'][-1] = new_r
                transitions[i]['discounts'] = new_gamma  # 假设需要存储新的折扣因子
                agent_group[i].update(transitions[i])

            step_list.append(env.step_count)
            win_list.append(env.explore_rate)


            if (i_episode + 1) % 100 == 0:
                pbar.set_postfix({
                    'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return': '%.3f' % np.mean(win_list[-100:])
                })
            pbar.update(1)
# 保存模型权重
# torch.save(agent.actor.state_dict(), 'weight/SPPO_actor_model2.pth')
# torch.save(agent.critic.state_dict(), 'weight/SPPO_critic_model2.pth')
step_array = np.array(step_list)
step_array = np.mean(step_array.reshape(-1, 100), axis=1)
win_array = np.array(win_list)
win_array = np.mean(win_array.reshape(-1, 100), axis=1)
episodes_list = np.arange(win_array.shape[0]) * 100
plt.plot(episodes_list, win_array)
plt.xlabel('Episodes')
plt.ylabel('Explore_rate')
plt.title('Independent PPO  on Exploration')
plt.savefig('result/IPPO.png')
plt.show()
plt.plot(episodes_list, step_array)
plt.xlabel('Episodes')
plt.ylabel('Step')
plt.title('Independent PPO  on Exploration')
plt.savefig('result/IPPO_step.png') 
plt.show()