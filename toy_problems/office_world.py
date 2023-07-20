import random, math, os
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

class OfficeWorld:
    def __init__(self, rf_id = 1):
        self._load_map()
        self.map_height, self.map_width = 12, 9
        self.rf = self._set_reward_function(rf_id)
        self.rf_pos = self.rf

    def _set_reward_function(self, rf_id):
        if rf_id == 0:
            rf = {
                'g' : {
                    'val' : 1, 'term' : True
                },
                'val' : 0, 'term' : False
            }
        elif rf_id == 1:
            rf = {
                'f' : {
                    'g' : {
                        'val' : 1, 'term' : True
                    },
                    'val' : 0, 'term' : False
                },
                'val' : 0, 'term' : False
            }
        elif rf_id == 2:
            rf = {
                'e' : {
                    'g' : {
                        'val' : 1, 'term' : True
                    },
                    'val' : 0, 'term' : False
                },
                'val': 0, 'term' : False
            }
        elif rf_id == 3:
            rf = {
                'f' : {
                    'e' : {
                        'g' : {
                            'val' : 1, 'term' : True
                        },
                        'val' : 0, 'term' : False
                    },
                    'val' : 0, 'term' : False
                },
                'e' : {
                    'f' : {
                        'g' : {
                            'val' : 1, 'term' : True
                        },
                        'val' : 0, 'term' : False
                    },
                    'val' : 0, 'term' : False
                },
                'val' : 0, 'term' : False
            }
        elif rf_id == 4:
            rf = {
                'a' : {
                    'b' : {
                        'c' : {
                            'd' : {
                                'val' : 1, 'term' : True
                            },
                            'val' : 0, 'term' : False
                        },
                        'val' : 0, 'term' : False
                    },
                    'val' : 0, 'term' : False
                },
                'val' : 0, 'term' : False
            }
        return rf


    def reset(self):
        self.agent = (2, 1)
        self.rf_pos = self.rf
        return self.agent

    def execute_action(self, a):
        """
        We execute 'action' in the game
        """
        x, y = self.agent
        self.agent = self._get_new_position(x, y, a)
        props = self._get_true_propositions()
        reward, done = self._get_reward(props)
        return reward, self.agent, done

    def _get_reward(self, prop):
        done = False
        if prop == 'n':
            reward = -1
            done = True
        else:
            if prop in self.rf_pos:
                self.rf_pos = self.rf_pos[prop]
            reward = self.rf_pos['val']
            done = self.rf_pos['term']
        return reward, done

    def _get_true_propositions(self):
        """
        Returns the string with the propositions that are True in this state
        """
        ret = ''
        if self.agent in self.objects:
            ret = self.objects[self.agent]
        return ret

    def _get_new_position(self, x, y, a):
        action = Actions(a)
        # executing action
        if (x, y, action) not in self.forbidden_transitions:
            if action == Actions.up:
                y += 1
            if action == Actions.down:
                y -= 1
            if action == Actions.left:
                x -= 1
            if action == Actions.right:
                x += 1
        return x, y

    def show(self):
        for y in range(8, -1, -1):
            if y % 3 == 2:
                for x in range(12):
                    if x % 3 == 0:
                        print("_", end="")
                        if 0 < x < 11:
                            print("_", end="")
                    if (x, y, Actions.up) in self.forbidden_transitions:
                        print("_", end="")
                    else:
                        print(" ", end="")
                print()
            for x in range(12):
                if (x, y, Actions.left) in self.forbidden_transitions:
                    print("|", end="")
                elif x % 3 == 0:
                    print(" ", end="")
                if (x, y) == self.agent:
                    print("A", end="")
                elif (x, y) in self.objects:
                    print(self.objects[(x, y)], end="")
                else:
                    print(" ", end="")
                if (x, y, Actions.right) in self.forbidden_transitions:
                    print("|", end="")
                elif x % 3 == 2:
                    print(" ", end="")
            print()
            if y % 3 == 0:
                for x in range(12):
                    if x % 3 == 0:
                        print("_", end="")
                        if 0 < x < 11:
                            print("_", end="")
                    if (x, y, Actions.down) in self.forbidden_transitions:
                        print("_", end="")
                    else:
                        print(" ", end="")
                print()

    def _load_map(self):
        # Creating the map
        self.objects = {}
        self.objects[(1, 1)] = "a"
        self.objects[(1, 7)] = "b"
        self.objects[(10, 7)] = "c"
        self.objects[(10, 1)] = "d"
        self.objects[(7, 4)] = "e"  # MAIL
        self.objects[(8, 2)] = "f"  # COFFEE
        self.objects[(3, 6)] = "f"  # COFFEE
        self.objects[(4, 4)] = "g"  # OFFICE
        self.objects[(4, 1)] = "n"  # PLANT
        self.objects[(7, 1)] = "n"  # PLANT
        self.objects[(4, 7)] = "n"  # PLANT
        self.objects[(7, 7)] = "n"  # PLANT
        self.objects[(1, 4)] = "n"  # PLANT
        self.objects[(10, 4)] = "n"  # PLANT
        # Adding walls
        self.forbidden_transitions = set()
        # general grid
        for x in range(12):
            for y in [0, 3, 6]:
                self.forbidden_transitions.add((x, y, Actions.down))
                self.forbidden_transitions.add((x, y + 2, Actions.up))
        for y in range(9):
            for x in [0, 3, 6, 9]:
                self.forbidden_transitions.add((x, y, Actions.left))
                self.forbidden_transitions.add((x + 2, y, Actions.right))
        # adding 'doors'
        for y in [1, 7]:
            for x in [2, 5, 8]:
                self.forbidden_transitions.remove((x, y, Actions.right))
                self.forbidden_transitions.remove((x + 1, y, Actions.left))
        for x in [1, 4, 7, 10]:
            self.forbidden_transitions.remove((x, 5, Actions.up))
            self.forbidden_transitions.remove((x, 6, Actions.down))
        for x in [1, 10]:
            self.forbidden_transitions.remove((x, 2, Actions.up))
            self.forbidden_transitions.remove((x, 3, Actions.down))
        # Adding the agent
        self.actions = [Actions.up.value, Actions.right.value, Actions.down.value, Actions.left.value]
