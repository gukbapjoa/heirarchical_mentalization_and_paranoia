from agents_models.abstract_agents import *


class SubIntentionalReceiver(SubIntentionalAgent):

    def __init__(self, actions, softmax_temp: float, threshold: Optional[float] = None):
        super().__init__(actions, softmax_temp, threshold)
        self.name = "DoM(-1)_receiver"

    def utility_function(self, action, observation):
        if self.threshold == 0:
            return np.array([1, 1])
        return (observation - self.threshold) * action

    def forward(self, action: Action, observation: Action, iteration_number=None):
        q_values = self.utility_function(self.potential_actions, observation.value)
        probabilities = self.softmax_transformation(q_values)
        return self.potential_actions, q_values, probabilities

    def update_nested_models(self, action=None, observation=None, iteration_number=None):
        pass # DoM(-1) agents do not have nested models to update