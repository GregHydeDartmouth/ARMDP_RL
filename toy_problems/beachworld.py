
class Beachworld():

    def __init__(self):
        x = 1
        self.states = ['shack','beach','exit']
        self.T = {'shack' : {'^': 'shack',
                             'v': 'exit',
                             'o': 'shack',
                             '<': 'beach',
                             '>': 'shack'},
                  'beach' : {'^': 'beach',
                             'v': 'beach',
                             'o': 'beach',
                             '<': 'beach',
                             '>': 'shack'}}
        self.sun_counter = 0
        self.position = None

    def get_state(self):
        return self.position

    def reward(self, state, action, next_state):
        reward = 0
        if state == 'beach' and action =='o' and next_state == 'beach':
            reward = 1 - 0.1 * self.sun_counter
            self.sun_counter += 1
        return round(reward,1)

    def step(self, action):
        next_state = self.T[self.position][action]
        reward = self.reward(self.position, action, next_state)
        self.position = next_state
        done = False
        if self.position == 'exit':
            done = True
        return reward, self.position, done

    def reset(self):
        self.sun_counter = 0
        self.position = 'beach'
        return self.position

