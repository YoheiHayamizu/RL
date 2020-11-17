from agent.AgentBasis import AgentBasisClass
import random


class UserAgent(AgentBasisClass):
    def __init__(self,
                 name="UserAgent",
                 gamma=0.99,
                 actions=None):
        super().__init__(name, actions, gamma)

    # Accessors

    def get_params(self):
        params = super().get_params()
        return params

    # Core

    def act(self, state):
        print("Select the index number of action from the followings.")
        tmp = list(self.__actions)
        for i, a in tmp:
            print("{0}: {1}".format(i, a), end=", ")
        print("input: ")
        action_num = int(input())
        action = tmp[action_num]

        self._number_of_steps += 1

        return action

    def reset(self):
        super().reset()
