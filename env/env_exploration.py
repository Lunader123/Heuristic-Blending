import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class GridWorld:
    def __init__(self, nrows, ncols, num_robots=3, initial_states=None, obstacles=None, max_steps=500):
        self.size = (nrows, ncols)  # Grid size
        self.max_steps = max_steps  # Max number of steps
        self.step_count = 0  # Step counter
        self.num_robots = num_robots
        self.grid = np.zeros(self.size)  # Initialize grid with unexplored cells (0)
        self.num_explored_cells = np.zeros(num_robots)  # To track exploration per robot
        self.initial_states = initial_states
        self.explore_rate = 0
        self.states = self.states = np.zeros(6)

        if obstacles is None:
            # Default no obstacles
            self.obstacles = np.random.randint(0, self.size[0], size=(5, 2))  # 障碍物的初始x, y坐标
            for obs in self.obstacles:
                self.grid[obs[0], obs[1]] = 1
        else:
            self.obstacles = obstacles
            for obs in self.obstacles:
                self.grid[obs[0], obs[1]] = 1  # Set obstacles in grid

        # self._initialize_grid()  # Initialize the grid with obstacles and initial exploration
        self.fig, self.ax = None, None  # For plotting

    # def _initialize_grid(self):
    #     # Mark the initial positions of the robots as explored
    #     for i, state in enumerate(self.initial_states):
    #         self._update_grid_exploration(state, i+1)

    def _update_grid_exploration(self, state, robot_idx):
        r, c = int(state[0]), int(state[1])
        if self.grid[r, c] == 0:  # Unexplored
            if robot_idx == 1:
                self.grid[r, c] = 0.25  # Robot A explores this cell
                self.num_explored_cells[0] += 1
            elif robot_idx == 2:
                self.grid[r, c] = 0.50  # Robot B explores this cell
                self.num_explored_cells[1] += 1
            elif robot_idx == 3:
                self.grid[r, c] = 0.75  # Robot C explores this cell
                self.num_explored_cells[2] += 1

    def _check_collision(self, new_state, idx):
        # Check for collisions with other robots or obstacles
        occupied = np.vstack([self.states[:idx], self.states[idx + 1:]])
        if not np.array_equal(self.obstacles, [-1]):
            occupied = np.vstack([occupied, self.obstacles])
        return any(np.all(new_state == occupied, axis=1))

    def reset(self):
        # 重置环境
        self.grid = np.zeros(self.size)  # 重置网格为未探索的状态
        self.step_count = 0  # 重置步数计数器
        self.explore_rate = 0
        self.num_explored_cells = np.zeros(3)  # 追踪每个智能体探索的格子数（假设有3个智能体）

        # 定义智能体的初始状态（x, y坐标）
        # initial_positions = np.array([[2, 2], [1, 4], [3, 1]])  # 智能体的初始x, y坐标
        initial_positions =  np.random.randint(0, self.size[0], size=(self.num_robots, 2))# 智能体的初始x, y坐标
        self.states = np.zeros((len(initial_positions), 6))  # 初始化每个智能体的状态为11维向量

        # 定义障碍物的位置，并在网格中标记障碍物
        # self.obstacles = np.array([[1, 1], [1, 5], [3, 1]])
        self.obstacles =  np.random.randint(0, self.size[0], size=(1, 2))# 障碍物的初始x, y坐标
        for obs in self.obstacles:
            self.grid[obs[0], obs[1]] = 1  # 将障碍物标记为1

        # 重置绘图的fig和ax
        self.fig, self.ax = None, None  # 用于绘图的初始化

        # 将智能体的初始位置标记为已探索，并更新网格的探索状态
        for i, position in enumerate(initial_positions):
            x, y = position  # 获取智能体的初始x, y坐标
            self.states[i][:2] = [x, y]  # 设置前两维为x和y坐标

            # 获取智能体周围9个格子的初始状态
            surrounding_state = self._get_surrounding_state(x, y)
            self.states[i][2:] = surrounding_state  # 将四个格子的状态填入剩下的维

            # 更新智能体对当前格子的探索状态
            self._update_grid_exploration(position, i + 1)  # 确保传递的是位置和智能体索引
        return self.states

    def _get_agent_idx_at_position(self, nx, ny):
        """
        查找位于(nx, ny)位置的智能体，并返回该智能体的索引。
        如果没有智能体在该位置，则返回None。
        """
        for idx, state in enumerate(self.states):
            agent_x, agent_y = state[0], state[1]  # 获取智能体的x, y坐标
            if agent_x == nx and agent_y == ny:
                return idx + 1  # 返回智能体的索引
        return None  # 如果没有找到智能体，返回None

    def _is_out_of_bounds(self, nx, ny):
        """
        检查给定的坐标 (nx, ny) 是否超出网格边界。
        如果坐标超出边界，则返回 True，否则返回 False。
        """
        height, width = self.size  # 获取网格的高度和宽度

        # 检查坐标是否在边界内
        if nx < 0 or nx >= height or ny < 0 or ny >= width:
            return True  # 坐标超出边界
        return False  # 坐标在边界内

    def _get_surrounding_state(self, x, y):
        """
        获取智能体(x, y)位置周围9个格子的状态并进行赋值：
        障碍物 = 1, 红色智能体 = 0.25, 蓝色智能体 = 0.75, 绿色智能体 = 0.5
        被红色智能体探索过 = 0.025, 被蓝色智能体探索过 = 0.075, 被绿色智能体探索过 = 0.05
        """
        surrounding_state = []
        # 遍历智能体周围的九个格子，包括当前位置
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = int(x + dx), int(y + dy)  # 计算相邻格子的坐标
            other_agent_idx = self._get_agent_idx_at_position(nx, ny)
            if self._is_out_of_bounds(nx, ny):
                # 如果超出边界，默认设为障碍物
                surrounding_state.append(1)
            elif (nx, ny) in self.obstacles:
                # 如果格子是障碍物
                surrounding_state.append(1)
            elif other_agent_idx:
                # 如果格子上有其他智能体，赋值不同的颜色
                if other_agent_idx == 1:  # 红色智能体
                    surrounding_state.append(0.25)
                elif other_agent_idx == 2:  # 蓝色智能体
                    surrounding_state.append(0.5)
                elif other_agent_idx == 3:  # 绿色智能体
                    surrounding_state.append(0.75)
            elif self.grid[nx, ny] == 0.25:  # 被红色智能体探索过
                surrounding_state.append(0.25)
            elif self.grid[nx, ny] == 0.5:  # 被绿色智能体探索过
                surrounding_state.append(0.5)
            elif self.grid[nx, ny] == 0.75:  # 被蓝色智能体探索过
                surrounding_state.append(0.75)
            else:
                # 如果格子是未探索的，默认为0
                surrounding_state.append(0)

        return surrounding_state

    def step(self, actions):
        rewards = np.zeros(len(self.states))  # 根据智能体数量初始化奖励数组
        next_states = self.states.copy()  # 下一步状态
        current_states = self.states.copy()  # 当前状态

        for idx, action in enumerate(actions):
            state = self.states[idx]
            x, y = state[0], state[1]  # 获取当前智能体的x、y坐标

            if action == 0:  # 等待动作
                rewards[idx] -= 0.2  # 等待惩罚
            else:
                # 获取根据动作更新后的新状态
                new_state = self._get_next_state(state, action)
                if np.array(new_state == state[:2]).all():
                    rewards[idx] -= 0.5
                new_x, new_y = new_state[0], new_state[1]

                # 检查移动是否合法
                if self._is_valid_move(new_state, idx):
                    next_states[idx][:2] = new_state[:2]  # 更新x, y坐标
                    rewards[idx] -= 0.1 # 移动惩罚
                    if self.grid[int(new_x), int(new_y)] == 0:  # 假设 0 表示未探索区域
                        rewards[idx] += 1 # 探索新区域的奖励

                    else:
                        rewards[idx] -= 0.5  # 重复探索的惩罚
                    # 更新智能体对周围格子的探索状态
                    self._update_grid_exploration(new_state, idx + 1)
                    # 更新周围9个格子的状态，假设 _get_surrounding_state() 会返回智能体周围的9个格子的状态
                    surrounding_state = self._get_surrounding_state(new_x, new_y)
                    next_states[idx][2:] = surrounding_state  # 更新状态向量的剩余部分（周围9个格子的状态）
                else:
                    rewards[idx] -= 5  # 非法移动惩罚

        # 更新所有智能体的状态
        self.states = next_states
        self.step_count += 1

        # 计算任务是否完成
        total_cells = np.prod(self.size) - len(self.obstacles)  # 总可探索格子数
        total_explored_cells = np.sum(self.num_explored_cells)  # 已探索格子数
        is_done = total_explored_cells >= total_cells  # 如果探索超过85%的格子，任务完成
        self.explore_rate = total_explored_cells / total_cells * 100
        # 如果任务完成，给予额外奖励
        if is_done:
            rewards += 200 * (self.num_explored_cells / total_cells)  # 根据已探索的格子比例给予奖励

        # 返回当前状态、下一步状态、奖励和任务完成标志
        return next_states, rewards, is_done

    def _get_next_state(self, state, action):
        # Action: 1 = up, 2 = down, 3 = left, 4 = right
        if action == 1 and state[0] > 0:
            return [state[0] - 1, state[1]]  # Move up
        elif action == 2 and state[0] < self.size[0] - 1:
            return [state[0] + 1, state[1]]  # Move down
        elif action == 3 and state[1] > 0:
            return [state[0], state[1] - 1]  # Move left
        elif action == 4 and state[1] < self.size[1] - 1:
            return [state[0], state[1] + 1]  # Move right
        else:
            return state  # Invalid move, stay in place

    def _is_valid_move(self, new_state, robot_idx):
        # Check for collisions with obstacles or other robots
        if self.grid[int(new_state[0]), int(new_state[1])] == 1:  # Obstacle
            return False
        for i, state in enumerate(self.states):
            if i != robot_idx and np.array_equal(state[:2], new_state[:2]):
                return False  # Collision with another robot
        return True

    def plot(self):
        # 定义自定义颜色映射
        colors = ['white', 'black', 'red', 'green', 'blue']  # 0.0, 1.0, 0.25, 0.5, 0.75
        cmap = ListedColormap(colors)

        # 手动将矩阵中的值映射到整数索引
        value_to_index = {
            0.0: 0,  # 0.0 -> white
            1.0: 1,  # 1.0 -> black
            0.25: 2,  # 0.25 -> red
            0.5: 3,  # 0.5 -> green
            0.75: 4  # 0.75 -> blue
        }

        # 将矩阵中的值转换为对应的索引
        mapped_grid = np.zeros_like(self.grid, dtype=int)

        for value, index in value_to_index.items():
            mapped_grid[self.grid == value] = index

        # 创建图形
        plt.imshow(mapped_grid, cmap=cmap, interpolation='nearest')
        plt.colorbar(label='Value')

        # 设置坐标轴
        plt.xticks(np.arange(self.grid.shape[1]), np.arange(1, self.grid.shape[1] + 1))
        plt.yticks(np.arange(self.grid.shape[0]), np.arange(1, self.grid.shape[0] + 1))
        plt.grid(False)

        # 绘制机器人
        robot_colors = ['red', 'green', 'blue']  # 不同机器人的颜色
        for i, state in enumerate(self.states):
            plt.scatter(state[1], state[0], color=robot_colors[i], s=200, edgecolor='black', label=f'Robot {i + 1}')

        # 添加图例
        plt.legend()
        plt.title('Grid Visualization with Robots')

        # 清除之前的图形以便动态更新
        plt.clf()
        plt.imshow(mapped_grid, cmap=cmap, interpolation='nearest')

        # 绘制机器人
        for i, state in enumerate(self.states):
            plt.scatter(state[1], state[0], color=robot_colors[i], s=200, edgecolor='black', label=f'Robot {i + 1}')

        plt.legend()
        plt.title('Grid Visualization with Robots')
        plt.pause(0.1)  # 暂停以更新显示

# if __name__ == "__main__":
#     world = GridWorld(15,15)
#     world.reset()
#     world.plot()
#     for _ in range(1000):  # Run simulation for 10 steps
#
#         actions = [np.random.randint(0, 5) for _ in range(3)]  # Random actions
#         nextstates, rewards, done = world.step(actions)
#
#         world.plot()
#         if done:
#             break
