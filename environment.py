class GridWorld:

    def __init__(self, grid_size=10, start_pos=(0, 0), goal_pos=(9, 9), obstacles=None):
        self.grid_size = grid_size
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.obstacles = obstacles if obstacles else []
        self.agent_pos = None
        self.actions = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

    def reset(self):
        self.agent_pos = list(self.start_pos)
        return tuple(self.agent_pos)

    def step(self, action):
        dx, dy = self.actions[action]
        new_x = self.agent_pos[0] + dx
        new_y = self.agent_pos[1] + dy

        if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
            if (new_x, new_y) not in self.obstacles:
                self.agent_pos = [new_x, new_y]

        reward = self._get_reward()
        done = self.is_done()
        return tuple(self.agent_pos), reward, done

    def _get_reward(self):
        if tuple(self.agent_pos) == self.goal_pos:
            return 100
        if tuple(self.agent_pos) in self.obstacles:
            return -10
        return -1

    def is_done(self):
        return tuple(self.agent_pos) == self.goal_pos

    def get_state(self):
        return tuple(self.agent_pos)
