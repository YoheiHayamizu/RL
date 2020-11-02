from agent.AgentBasis import AgentBasisClass
import random


class QLearningAgent(AgentBasisClass):
    def __init__(self,
                 name="RandomAgent",
                 gamma=0.99,
                 actions=None):
        super().__init__(name, actions, gamma)

    # Accessors

    def get_params(self):
        params = super().get_params()
        return params

    # Core

    def act(self, state):
        action = random.choice(self.actions)

        self.number_of_steps += 1

        return action

    def reset(self):
        super().reset()
