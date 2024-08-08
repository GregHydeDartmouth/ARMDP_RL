import copy
from enum import Enum
"""
Enum with the actions that the agent can execute
"""
class Actions(Enum):
    up    = 0 # move up
    right = 1 # move right
    down  = 2 # move down
    left  = 3 # move left
    none  = 4 # none or pick

class BreakfastWorld:
    def __init__(self, cumulative_only = False, rm=1):
        self._load_map()
        self.map_height, self.map_width = 6, 4
        self.rf, self.rf_position = self._get_reward_function(rm=rm)
        self.cumulative_only = cumulative_only
        self.cumulative_reward = 0

    def _get_reward_function(self, rm):
        if rm == 1:
            rf = {
                'c' : {
                    'val' : -0.1, 'term' : False,
                    'e' : {
                        'val' : 1, 'term' : False,
                        'l' : {
                            'val' : 0, 'term' : True
                        }
                    },
                    'c' : {
                        'val' : -0.3, 'term' : False,
                        'e' : {
                            'val' : 2, 'term' : False,
                            'l' : {
                                'val' : 0, 'term' : True
                            }
                        },
                    }
                }
            }
            rf_position = copy.deepcopy(rf)
            return rf, rf_position
        if rm == 2:
            rf = {
                'c' : {
                    'val' : -0.1, 'term' : False,
                    'e' : {
                        'val' : 1, 'term' : False,
                        'w' : {
                            'val' : -0.25, 'term' : False,
                            'l' : {
                                'val' : 0, 'term' : True
                            }
                        }
                    },
                    'c' : {
                        'val' : -0.3, 'term' : False,
                        'e' : {
                            'val' : 2, 'term' : False,
                            'w' : {
                                'val' : -0.5, 'term' : False,
                                'l' : {
                                    'val' : 0, 'term' : True
                                }
                            }
                        }
                    }
                }
            }
            rf_position = copy.deepcopy(rf)
            return rf, rf_position

    def reset(self):
        self.agent = (0, 0)
        self.rf_position = copy.deepcopy(self.rf)
        self.cumulative_reward = 0
        return self.agent

    def _get_observable_proposition(self):
        """
        Returns the string with the propositions that are True in this state
        """
        if self.agent in self.objects:
            return self.objects[self.agent]
        else:
            return '*'

    def execute_action(self, a):
        """
        We execute 'action' in the game
        """
        x, y = self.agent
        self.agent = self._get_new_position(x, y, a)
        observable_prop = self._get_observable_proposition()
        reward, done = self._get_reward(observable_prop)
        return reward, self.agent, done

    def _get_reward(self, observable_prop):
        reward = 0
        done = False
        if observable_prop in self.rf_position:
            self.rf_position = self.rf_position[observable_prop]
            reward += self.rf_position['val']
            done = self.rf_position['term']
            self.cumulative_reward += reward
        if self.cumulative_only:
            if done:
                return round(self.cumulative_reward, 4), done
            return 0, done
        else:
            if done:
                return round(self.cumulative_reward, 4), done
            return round(reward, 4), done

    def _get_new_position(self, x, y, a):
        action = Actions(a)
        # executing action
        if (x, y, action) not in self.forbidden_transitions:
            _x = x
            _y = y
            if action == Actions.up:
                _y += 1
            if action == Actions.down:
                _y -= 1
            if action == Actions.left:
                _x -= 1
            if action == Actions.right:
                _x += 1
            if (_x, _y) in self.solid_objects:
                return x, y
            return _x, _y
        return x, y

    def show(self):
        for y in range(6):
            for x in range(4):
                symbol = ' '
                if (x, 5-y) in self.objects:
                    symbol = self.objects[(x, 5-y)]
                if (x, 5-y) == self.agent:
                    symbol = 'A'
                if (x, 5-y) in self.solid_objects:
                    symbol = 'X'
                print('[{}]'.format(symbol), end='')
            print()

    def _load_map(self):
        # Creating the map
        self.objects = {}
        self.objects[(0, 0)] = 'l'
        self.objects[(2, 2)] = 'e'
        self.objects[(2, 4)] = 'w'
        self.objects[(1, 4)] = 'c'
 

        # objects like beds and counter that are removed from state space
        self.solid_objects = {(3, 1), (3, 2), (3, 3), (3, 4), (3, 5),
                              (2, 5), (1, 5), (0, 5), (0, 4), (0, 3)}


        self.forbidden_transitions = set()
        # outer-walls
        for x in range(0, 4):
            self.forbidden_transitions.add((x, 0, Actions.down))
            self.forbidden_transitions.add((x, 5, Actions.up))
        for y in range(0, 6):
            self.forbidden_transitions.add((0, y, Actions.left))
            self.forbidden_transitions.add((3, y, Actions.right))

        # inner walls
        self.forbidden_transitions.add((2, 3, Actions.down))
        self.forbidden_transitions.add((2, 2, Actions.up))

        # Adding the agent
        self.actions = [Actions.up.value, Actions.right.value, Actions.down.value, Actions.left.value]
