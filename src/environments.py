import numpy as np


class L9Env:
    def __init__(self):
        self.route = None
        self.current_state = None
        self.rewards_new = None
        self.ending_state = None
        self.rewards = None
        self.num_states = 9
        self.location_to_state = {
            'L1': 0, 'L2': 1, 'L3': 2,
            'L4': 3, 'L5': 4, 'L6': 5,
            'L7': 6, 'L8': 7, 'L9': 8}

        self.state_to_location = dict((state, location) for location, state in self.location_to_state.items())

        self.rewards_base = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0],
                                      [1, 0, 1, 0, 0, 0, 0, 0, 0],
                                      [0, 1, 0, 0, 0, 1, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 1, 0, 0],
                                      [0, 1, 0, 0, 0, 0, 0, 1, 0],
                                      [0, 0, 1, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 1, 0, 0, 0, 1, 0],
                                      [0, 0, 0, 0, 1, 0, 1, 0, 1],
                                      [0, 0, 0, 0, 0, 0, 0, 1, 0]])

    def reset(self, end_location):
        self.rewards = np.copy(self.rewards_base)
        self.ending_state = self.location_to_state[end_location]
        self.rewards[self.ending_state, self.ending_state] = 999
        self.current_state = np.random.randint(0, 9)
        self.route = []

    def get_state(self):
        return self.current_state

    def get_reward(self, start, stop):
        return self.rewards[start, stop]

    def update(self, action):
        self.current_state = action
