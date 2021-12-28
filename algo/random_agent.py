from random import randint


class RandomAgent(object):
    def __init__(self, key, action_space):
        self.key = key
        self.action_space = action_space

    def choose_action(self, observation):
        actions = []
        for i in range(len(self.action_space)):
            action = []
            low = self.action_space[i].low
            high = self.action_space[i].high
            for j in range(self.action_space[i].shape[0]):
                act = randint(low[j], high[j])
                action.append(act)
            actions.append(action)

        return actions
