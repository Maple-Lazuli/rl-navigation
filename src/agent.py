import numpy as np
from environments import L9Env


class Agent:
    def __init__(self, env, actions=9, gamma=0.75, alpha=0.9):
        self.gamma = gamma
        self.alpha = alpha
        self.actions = [i for i in range(0, actions)]
        self.Q = np.array(np.zeros([actions, actions]))
        self.actions = actions
        self.env = env

    def train(self, end_location, episodes=1000):
        for i in range(episodes):
            self.env.reset(end_location)
            current_state = self.env.get_state()
            playable_actions = []

            for j in range(self.actions):
                if self.env.get_reward(current_state, j) > 0:
                    playable_actions.append(j)

            self.env.update(np.random.choice(playable_actions))
            next_state = self.env.get_state()

            TD = self.env.get_reward(current_state, next_state)
            TD += self.gamma * self.Q[next_state, np.argmax(self.Q[next_state,])] - \
                  self.Q[current_state, next_state]

            self.Q[current_state, next_state] += self.alpha * TD

    def get_optimal_route(self, start, end):
        route = [start]
        next = start

        while next != end:
            starting_state = self.env.location_to_state[start]

            next_state = np.argmax(self.Q[starting_state,])

            next = self.env.state_to_location[next_state]
            route.append(next)

            start = next

        return route


if __name__ == '__main__':
    env = L9Env()

    agent = Agent(env=env)

    agent.train(end_location='L1')

    route = agent.get_optimal_route('L9', 'L1')

    print(route)
