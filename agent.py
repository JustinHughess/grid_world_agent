import random


class QLearningAgent:

    def __init__(self, n_actions=4, learning_rate=0.1, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        q_values = [self.get_q_value(state, action) for action in range(self.n_actions)]
        max_q = max(q_values)
        best_actions = [action for action in range(self.n_actions) if q_values[action] == max_q]
        return random.choice(best_actions)

    def update_q_value(self, state, action, reward, next_state):
        current_q = self.get_q_value(state, action)
        max_next_q = max([self.get_q_value(next_state, a) for a in range(self.n_actions)])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[(state, action)] = new_q

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_best_action(self, state):
        q_values = [self.get_q_value(state, action) for action in range(self.n_actions)]
        max_q = max(q_values)
        best_actions = [action for action in range(self.n_actions) if q_values[action] == max_q]
        return random.choice(best_actions)
