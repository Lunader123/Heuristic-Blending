import torch
import numpy as np
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SumTree(object):  # sumtree的定义
    """
    Story data with its priority in the tree.
    Tree structure and array storage:

    Tree index:
         0         -> storing priority sum
        / \
      1     2
     / \   / \
    3   4 5   6    -> storing priority for transitions

    Array type for storing:
    [0,1,2,3,4,5,6]
    """

    def __init__(self, buffer_capacity):
        self.buffer_capacity = buffer_capacity  # buffer的容量
        self.tree_capacity = 2 * buffer_capacity - 1  # sum_tree的容量
        self.tree = np.zeros(self.tree_capacity)

    def update_priority(self, data_index, priority):
        ''' Update the priority for one transition according to its index in buffer '''
        # data_index表示当前数据在buffer中的index
        # tree_index表示当前数据在sum_tree中的index
        tree_index = data_index + self.buffer_capacity - 1  # 把当前数据在buffer中的index转换为在sum_tree中的index
        change = priority - self.tree[tree_index]  # 当前数据的priority的改变量
        self.tree[tree_index] = priority  # 更新树的最后一层叶子节点的优先级
        # then propagate the change through the tree
        while tree_index != 0:  # 更新上层节点的优先级，一直传播到最顶端
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def prioritized_sample(self, N, batch_size, beta):
        ''' sample a batch of index and normlized IS weight according to priorites '''
        batch_index = np.zeros(batch_size, dtype=np.uint32)
        IS_weight = torch.zeros(batch_size, dtype=torch.float32)
        segment = self.priority_sum / batch_size  # 把[0,priority_sum]等分成batch_size个区间，在每个区间均匀采样一个数
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            v = np.random.uniform(a, b)
            buffer_index, priority = self._get_index(v)
            batch_index[i] = buffer_index
            prob = priority / self.priority_sum  # 当前数据被采样的概率
            IS_weight[i] = (N * prob) ** (-beta)
        Normed_IS_weight = IS_weight / IS_weight.max()  # normalization

        return batch_index, Normed_IS_weight

    def _get_index(self, v):
        ''' sample a index '''
        parent_idx = 0  # 从树的顶端开始
        while True:
            child_left_idx = 2 * parent_idx + 1  # 父节点下方的左右两个子节点的index
            child_right_idx = child_left_idx + 1
            if child_left_idx >= self.tree_capacity:  # reach bottom, end search
                tree_index = parent_idx  # tree_index表示采样到的数据在sum_tree中的index
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[child_left_idx]:
                    parent_idx = child_left_idx
                else:
                    v -= self.tree[child_left_idx]
                    parent_idx = child_right_idx

        data_index = tree_index - self.buffer_capacity + 1  # tree_index->data_index
        return data_index, self.tree[tree_index]  # 返回采样到的data在buffer中的index,以及相对应的priority

    @property
    def priority_sum(self):
        return self.tree[0]  # 树的顶端保存了所有priority之和

    @property
    def priority_max(self):
        return self.tree[self.buffer_capacity - 1:].max()  # 树的最后一层叶节点，保存的才是每个数据对应的priority


class ReplayBuffer1(object):  # 正常经验回放区
    def __init__(self, args):
        self.N = args.N  # The number of agents
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.count = 0
        self.current_size = 0
        self.buffer_obs_n, self.buffer_a_n, self.buffer_r_n, self.buffer_s_next_n, self.buffer_done_n = [], [], [], [], []
        for agent_id in range(self.N):
            self.buffer_obs_n.append(np.empty((self.buffer_size, args.obs_dim_n[agent_id])))
            self.buffer_a_n.append(np.empty((self.buffer_size, args.action_dim_n[agent_id])))
            self.buffer_r_n.append(np.empty((self.buffer_size, 1)))
            self.buffer_s_next_n.append(np.empty((self.buffer_size, args.obs_dim_n[agent_id])))
            self.buffer_done_n.append(np.empty((self.buffer_size, 1)))

    def store_transition(self, obs_n, a_n, r_n, obs_next_n, done_n):
        for agent_id in range(self.N):
            self.buffer_obs_n[agent_id][self.count] = obs_n[agent_id]
            self.buffer_a_n[agent_id][self.count] = a_n[agent_id]
            self.buffer_r_n[agent_id][self.count] = r_n[agent_id]
            self.buffer_s_next_n[agent_id][self.count] = obs_next_n[agent_id]
            self.buffer_done_n[agent_id][self.count] = done_n[agent_id]
        self.count = (self.count + 1) % self.buffer_size  # When the 'count' reaches max_size, it will be reset to 0.
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def sample(self, ):
        index = np.random.choice(self.current_size, size=self.batch_size, replace=False)
        batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = [], [], [], [], []
        for agent_id in range(self.N):
            batch_obs_n.append(torch.tensor(self.buffer_obs_n[agent_id][index], dtype=torch.float).to(device))
            batch_a_n.append(torch.tensor(self.buffer_a_n[agent_id][index], dtype=torch.float).to(device))
            batch_r_n.append(torch.tensor(self.buffer_r_n[agent_id][index], dtype=torch.float).to(device))
            batch_obs_next_n.append(torch.tensor(self.buffer_s_next_n[agent_id][index], dtype=torch.float).to(device))
            batch_done_n.append(torch.tensor(self.buffer_done_n[agent_id][index], dtype=torch.float).to(device))

        return batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n


class ReplayBuffer2(object):  # 成本函数经验回放区
    def __init__(self, args):
        self.N = args.N  # The number of agents
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.count = 0
        self.current_size = 0

        self.buffer_obs_n, self.buffer_a_n, self.buffer_r_n, self.buffer_s_next_n, self.buffer_done_n, self.buffer_c_n ,self.buffer_c_n1 ,self.buffer_c_n2 = [], [], [], [], [], [],[],[]
        for agent_id in range(self.N):
            self.buffer_obs_n.append(np.empty((self.buffer_size, args.obs_dim_n[agent_id])))
            self.buffer_a_n.append(np.empty((self.buffer_size, args.action_dim_n[agent_id])))
            self.buffer_r_n.append(np.empty((self.buffer_size, 1)))
            self.buffer_s_next_n.append(np.empty((self.buffer_size, args.obs_dim_n[agent_id])))
            self.buffer_done_n.append(np.empty((self.buffer_size, 1)))
            self.buffer_c_n.append(np.empty((self.buffer_size, 1)))
            self.buffer_c_n1.append(np.empty((self.buffer_size, 1)))
            self.buffer_c_n2.append(np.empty((self.buffer_size, 1)))

    def store_transition(self, obs_n, a_n, r_n, obs_next_n, done_n, c_n,c_n1,c_n2):
        for agent_id in range(self.N):
            self.buffer_obs_n[agent_id][self.count] = obs_n[agent_id]
            self.buffer_a_n[agent_id][self.count] = a_n[agent_id]
            self.buffer_r_n[agent_id][self.count] = r_n[agent_id]
            self.buffer_s_next_n[agent_id][self.count] = obs_next_n[agent_id]
            self.buffer_done_n[agent_id][self.count] = done_n[agent_id]
            self.buffer_c_n[agent_id][self.count] = c_n[agent_id]
            self.buffer_c_n1[agent_id][self.count] = c_n1[agent_id]
            self.buffer_c_n2[agent_id][self.count] = c_n2[agent_id]

        self.count = (self.count + 1) % self.buffer_size  # When the 'count' reaches max_size, it will be reset to 0.
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def sample(self, ):
        index = np.random.choice(self.current_size, size=self.batch_size, replace=False)
        batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n, batch_c_n , batch_c_n1 , batch_c_n2 = [], [], [], [], [], [],[],[]
        for agent_id in range(self.N):
            batch_obs_n.append(torch.tensor(self.buffer_obs_n[agent_id][index], dtype=torch.float).to(device))
            batch_a_n.append(torch.tensor(self.buffer_a_n[agent_id][index], dtype=torch.float).to(device))
            batch_r_n.append(torch.tensor(self.buffer_r_n[agent_id][index], dtype=torch.float).to(device))
            batch_obs_next_n.append(torch.tensor(self.buffer_s_next_n[agent_id][index], dtype=torch.float).to(device))
            batch_done_n.append(torch.tensor(self.buffer_done_n[agent_id][index], dtype=torch.float).to(device))
            batch_c_n.append(torch.tensor(self.buffer_c_n[agent_id][index], dtype=torch.float).to(device))
            batch_c_n1.append(torch.tensor(self.buffer_c_n1[agent_id][index], dtype=torch.float).to(device))
            batch_c_n2.append(torch.tensor(self.buffer_c_n2[agent_id][index], dtype=torch.float).to(device))
        return batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n, batch_c_n, batch_c_n1, batch_c_n2


class ReplayBuffer_ppo:  #mappo正常经验回放区
    def __init__(self, args):
        self.N = args.N
        self.buffer_size = args.buffer_size
        self.s, self.a, self.a_logprob, self.r, self.s_, self.done = [], [], [], [], [], []
        for agent_id in range(self.N):
            self.s.append(np.zeros((args.buffer_size, args.obs_dim_n[agent_id])))
            self.a.append(np.zeros((args.buffer_size, args.action_dim_n[agent_id])))
            self.a_logprob.append(np.zeros((args.buffer_size, args.action_dim_n[agent_id])))
            self.r.append(np.zeros((args.buffer_size, 1)))
            self.s_.append(np.zeros((args.buffer_size, args.obs_dim_n[agent_id])))
            self.done.append(np.zeros((args.buffer_size, 1)))

        self.count = 0

    def store_transition(self, s, a, a_logprob, r, s_, done):
        for agent_id in range(self.N):
            self.s[agent_id][self.count] = s[agent_id]
            self.a[agent_id][self.count] = a[agent_id]
            self.a_logprob[agent_id][self.count] = a_logprob[agent_id]
            self.r[agent_id][self.count] = r[agent_id]
            self.s_[agent_id][self.count] = s_[agent_id]
            self.done[agent_id][self.count] = done[agent_id]
        self.count += 1

    def numpy_to_tensor(self):
        batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n, batch_a_logprob_n = [], [], [], [], [], []
        for agent_id in range(self.N):
            batch_obs_n.append(torch.tensor(self.s[agent_id], dtype=torch.float).to(device))
            batch_a_n.append(torch.tensor(self.a[agent_id], dtype=torch.float).to(device))
            batch_r_n.append(torch.tensor(self.r[agent_id], dtype=torch.float).to(device))
            batch_obs_next_n.append(torch.tensor(self.s_[agent_id], dtype=torch.float).to(device))
            batch_done_n.append(torch.tensor(self.done[agent_id], dtype=torch.float).to(device))
            batch_a_logprob_n.append(torch.tensor(self.a_logprob[agent_id], dtype=torch.float).to(device))

        return batch_obs_n, batch_a_n,batch_a_logprob_n, batch_r_n, batch_obs_next_n, batch_done_n

class ReplayBuffer_ppo_cost:  #mappo成本经验回放区
    def __init__(self, args):
        self.N = args.N
        self.buffer_size = args.buffer_size
        self.s, self.a, self.a_logprob, self.r, self.s_, self.done ,self.c,self.c1,self.c2= [], [], [], [], [], [],[],[],[]
        for agent_id in range(self.N):
            self.s.append(np.zeros((args.buffer_size, args.obs_dim_n[agent_id])))
            self.a.append(np.zeros((args.buffer_size, args.action_dim_n[agent_id])))
            self.a_logprob.append(np.zeros((args.buffer_size, args.action_dim_n[agent_id])))
            self.r.append(np.zeros((args.buffer_size, 1)))
            self.c.append(np.zeros((args.buffer_size, 1)))
            self.c1.append(np.zeros((args.buffer_size, 1)))
            self.c2.append(np.zeros((args.buffer_size, 1)))
            self.s_.append(np.zeros((args.buffer_size, args.obs_dim_n[agent_id])))
            self.done.append(np.zeros((args.buffer_size, 1)))

        self.count = 0

    def store_transition(self, s, a, a_logprob, r, s_, done,c,c1,c2):
        for agent_id in range(self.N):
            self.s[agent_id][self.count] = s[agent_id]
            self.a[agent_id][self.count] = a[agent_id]
            self.a_logprob[agent_id][self.count] = a_logprob[agent_id]
            self.r[agent_id][self.count] = r[agent_id]
            self.c[agent_id][self.count] = c[agent_id]
            self.c1[agent_id][self.count] = c1[agent_id]
            self.c2[agent_id][self.count] = c2[agent_id]
            self.s_[agent_id][self.count] = s_[agent_id]
            self.done[agent_id][self.count] = done[agent_id]
        self.count += 1

    def numpy_to_tensor(self):
        batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n, batch_a_logprob_n, batch_c_n, batch_c1_n, batch_c2_n = [], [], [], [], [], [],[],[],[]
        for agent_id in range(self.N):
            batch_obs_n.append(torch.tensor(self.s[agent_id], dtype=torch.float).to(device))
            batch_a_n.append(torch.tensor(self.a[agent_id], dtype=torch.float).to(device))
            batch_r_n.append(torch.tensor(self.r[agent_id], dtype=torch.float).to(device))
            batch_c_n.append(torch.tensor(self.c[agent_id], dtype=torch.float).to(device))
            batch_c1_n.append(torch.tensor(self.c1[agent_id], dtype=torch.float).to(device))
            batch_c2_n.append(torch.tensor(self.c2[agent_id], dtype=torch.float).to(device))
            batch_obs_next_n.append(torch.tensor(self.s_[agent_id], dtype=torch.float).to(device))
            batch_done_n.append(torch.tensor(self.done[agent_id], dtype=torch.float).to(device))
            batch_a_logprob_n.append(torch.tensor(self.a_logprob[agent_id], dtype=torch.float).to(device))

        return batch_obs_n, batch_a_n,batch_a_logprob_n, batch_r_n, batch_obs_next_n, batch_done_n, batch_c_n, batch_c1_n, batch_c2_n
class ReplayBuffer(object):  # 优先经验回放区
    def __init__(self, args):
        self.ptr = 0
        self.size = 0
        self.N = args.N  # The number of agents
        self.buffer_size = args.buffer_size
        max_size = int(self.buffer_size)
        self.batch_size = args.batch_size
        self.count = 0
        self.current_size = 0
        self.alpha = 0.6
        self.beta = 0.4
        self.device = device
        self.sum_tree = []
        for i in range(self.N):
            self.sum_tree.append(SumTree(max_size))
        self.buffer_obs_n, self.buffer_a_n, self.buffer_r_n, self.buffer_s_next_n, self.buffer_done_n = [], [], [], [], []
        for agent_id in range(self.N):
            self.buffer_obs_n.append(np.empty((self.buffer_size, args.obs_dim_n[agent_id])))
            self.buffer_a_n.append(np.empty((self.buffer_size, args.action_dim_n[agent_id])))
            self.buffer_r_n.append(np.empty((self.buffer_size, 1)))
            self.buffer_s_next_n.append(np.empty((self.buffer_size, args.obs_dim_n[agent_id])))
            self.buffer_done_n.append(np.empty((self.buffer_size, 1)))

    def store_transition(self, obs_n, a_n, r_n, obs_next_n, done_n):
        for agent_id in range(self.N):
            self.buffer_obs_n[agent_id][self.count] = obs_n[agent_id]
            self.buffer_a_n[agent_id][self.count] = a_n[agent_id]
            self.buffer_r_n[agent_id][self.count] = r_n[agent_id]
            self.buffer_s_next_n[agent_id][self.count] = obs_next_n[agent_id]
            self.buffer_done_n[agent_id][self.count] = done_n[agent_id]
            # 如果是第一条经验，初始化优先级为1.0；否则，对于新存入的经验，指定为当前最大的优先级
            priority = 1.0 if self.current_size == 0 else self.sum_tree[agent_id].priority_max
            self.sum_tree[agent_id].update_priority(data_index=self.count, priority=priority)  # 更新当前经验在sum_tree中的优先级
        self.count = (self.count + 1) % self.buffer_size  # When the 'count' reaches max_size, it will be reset to 0.
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def sample(self, ):
        # index = np.random.choice(self.current_size, size=self.batch_size, replace=False)
        batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = [], [], [], [], []
        indexs = []
        Normed_IS_weights = []
        for agent_id in range(self.N):
            index, Normed_IS_weight = self.sum_tree[agent_id].prioritized_sample(N=self.current_size,
                                                                                 batch_size=self.batch_size,
                                                                                 beta=self.beta)
            indexs.append(index)
            Normed_IS_weights.append(Normed_IS_weight)
            batch_obs_n.append(torch.tensor(self.buffer_obs_n[agent_id][index], dtype=torch.float).to(device))
            batch_a_n.append(torch.tensor(self.buffer_a_n[agent_id][index], dtype=torch.float))
            batch_r_n.append(torch.tensor(self.buffer_r_n[agent_id][index], dtype=torch.float))
            batch_obs_next_n.append(torch.tensor(self.buffer_s_next_n[agent_id][index], dtype=torch.float))
            batch_done_n.append(torch.tensor(self.buffer_done_n[agent_id][index], dtype=torch.float))

        return batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n, indexs, Normed_IS_weights

    def update_batch_priorities(self, batch_index, td_errors, agent_id):  # 根据传入的td_error，更新batch_index所对应数据的priorities
        priorities = (np.abs(td_errors) + 0.01) ** self.alpha
        for index, priority in zip(batch_index, priorities):
            self.sum_tree[agent_id].update_priority(data_index=index, priority=priority)


class PrioritizedReplayBuffer(object):
    def __init__(self, opt):
        self.ptr = 0
        self.size = 0
        max_size = int(opt.buffer_size)
        self.state = np.zeros((max_size, opt.state_dim))
        self.action = np.zeros((max_size, 1))
        self.reward = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, opt.state_dim))
        self.dw = np.zeros((max_size, 1))
        self.max_size = max_size
        self.sum_tree = SumTree(max_size)
        self.alpha = opt.alpha
        self.beta = opt.beta_init
        self.device = device

    def add(self, state, action, reward, next_state, dw):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.dw[self.ptr] = dw  # 0,0,0，...，1
        # 如果是第一条经验，初始化优先级为1.0；否则，对于新存入的经验，指定为当前最大的优先级
        priority = 1.0 if self.size == 0 else self.sum_tree.priority_max
        self.sum_tree.update_priority(data_index=self.ptr, priority=priority)  # 更新当前经验在sum_tree中的优先级
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind, Normed_IS_weight = self.sum_tree.prioritized_sample(N=self.size, batch_size=batch_size, beta=self.beta)
        return (
            torch.tensor(self.state[ind], dtype=torch.float32).to(self.device),
            torch.tensor(self.action[ind], dtype=torch.long).to(self.device),
            torch.tensor(self.reward[ind], dtype=torch.float32).to(self.device),
            torch.tensor(self.next_state[ind], dtype=torch.float32).to(self.device),
            torch.tensor(self.dw[ind], dtype=torch.float32).to(self.device),
            ind,
            Normed_IS_weight.to(self.device)  # shape：(batch_size,)
        )

    def update_batch_priorities(self, batch_index, td_errors):  # 根据传入的td_error，更新batch_index所对应数据的priorities
        priorities = (np.abs(td_errors) + 0.01) ** self.alpha
        for index, priority in zip(batch_index, priorities):
            self.sum_tree.update_priority(data_index=index, priority=priority)
