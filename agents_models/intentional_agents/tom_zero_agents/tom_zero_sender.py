from sympy import sequence

from agents_models.subintentional_agents.subintentional_receiver import *
import functools


class DoMZeroSenderBelief(DoMZeroBelief):

    def __init__(self, support, zero_level_belief, opponent_model: SubIntentionalAgent, history: History):
        super().__init__(support, zero_level_belief, opponent_model, history)

    def compute_likelihood(self, action, observation, prior, iteration_number=None, nested=False):
        """
        Compute observation likelihood given opponent's type and last action
        :param iteration_number:
        :param action:
        :param observation:
        :param prior:
        :return:
        """
        if observation is None or observation.value is None:
            return np.ones_like(prior) / len(prior)  # flat/uniform prior at trial 0
        last_observation = self.history.get_last_observation()
        offer_likelihood = np.empty_like(prior)
        original_threshold = self.opponent_model.threshold
        for i in range(len(self.prior_belief[:, 0])):
            theta = self.prior_belief[:, 0][i]
            self.opponent_model.threshold = theta
            possible_opponent_actions, opponent_q_values, probabilities = \
                self.opponent_model.forward(last_observation, action, iteration_number)
            # If the observation is not in the feasible action set then it singles theta hat:
            observation_probability = probabilities[np.where(possible_opponent_actions == observation.value)]
            offer_likelihood[i] = observation_probability
        self.opponent_model.threshold = original_threshold
        return offer_likelihood


class DoMZeroSenderEnvironmentModel(DoMZeroEnvironmentModel):

    def __init__(self, opponent_model: SubIntentionalAgent, reward_function,
                 actions: np.array,
                 belief_distribution):
        super().__init__(opponent_model, reward_function, actions, belief_distribution)


class DoMZeroSenderExplorationPolicy(DoMZeroExplorationPolicy):

    def __init__(self, actions, reward_function, exploration_bonus, belief: np.array, type_support: np.array):
        super().__init__(actions, reward_function, exploration_bonus, belief, type_support)

    def sample(self, interactive_state: InteractiveState, last_action: float, observation: bool, iteration_number: int):
        # if the last offer was accepted - we can offer less (if we can)
        if observation:
            weights = np.array([last_action >= self.actions]) * 0.5
        # if the last offer was rejected - we should offer more (if we can)
        else:
            if last_action < np.max(self.actions):
                weights = np.array([last_action < self.actions]) * 0.5
            else:
                weights = 0.0
        acceptance_odds = np.array([x >= 1-self.actions for x in self.support[1:]]).T
        acceptance_odds = np.c_[np.repeat(True, len(self.actions)), acceptance_odds]
        current_beliefs = interactive_state.get_nested_belief
        acceptance_probability = np.multiply(current_beliefs, acceptance_odds).sum(axis=1)
        expected_reward_from_offer = self.reward_function(self.actions, True) * acceptance_probability
        optimal_action_idx = np.argmax(expected_reward_from_offer)
        optimal_action = self.actions[optimal_action_idx]
        q_value = expected_reward_from_offer[optimal_action_idx]
        return Action(optimal_action, False), q_value

    def init_q_values(self, observation: Action, *args):
        is_irritated = 1
        if len(args) > 0:
            if len(args[0]) > 0:
                persona = [x.get_persona[1] for x in args[0].values()]
                is_irritated = 1 - np.any(persona)
        reward_from_action = self.reward_function(self.actions, True)
        acceptance_probability = self.acceptance_probability_per_type(self.support)
        initial_qvalues = np.multiply(reward_from_action, acceptance_probability)
        return initial_qvalues * is_irritated

    def acceptance_probability_per_type(self, belief):
        accept_reject_by_type = self.actions[:, np.newaxis] >= self.support
        probability_by_action = np.sum(accept_reject_by_type/accept_reject_by_type.shape[1], axis=1)
        return probability_by_action


class DoMZeroSenderSolver(DoMZeroEnvironmentModel):
    def __init__(self, actions, belief_distribution: DoMZeroBelief, opponent_model: SubIntentionalAgent,
                 reward_function, planning_horizon, discount_factor):
        super().__init__(opponent_model, reward_function, actions, belief_distribution)
        self.actions = actions
        self.belief = belief_distribution
        self.opponent_model = opponent_model
        self.utility_function = reward_function
        self.planning_horizon = planning_horizon
        self.discount_factor = discount_factor
        self.action_node = None
        self.surrogate_actions = [Action(value, False) for value in self.actions]
        self.name = "tree_search"
        self.tree = []

    def plan(self, action, observation, iteration_number, update_belief):
        # Belief update via IRL
        action_length = len(self.belief.history.actions)
        observation_length = len(self.belief.history.observations)
        if update_belief:
            self.belief.update_distribution(action, observation, iteration_number)
        # Recursive planning_tree spanning
        q_values_array = []
        for threshold in self.belief.belief_distribution[:, 0]:
            # Reset nested model
            self.reset_persona(threshold, action_length, observation_length,
                               self.opponent_model.belief, iteration_number)
            future_values = functools.partial(self.compute_expected_value_from_offer, observation=observation,
                                              opponent_model=self.opponent_model,
                                              iteration_number=iteration_number)
            q_values = list(map(future_values, self.surrogate_actions))
            q_values_array.append(q_values)
        weighted_q_values = self.belief.belief_distribution[:, -1] @ np.array(q_values_array)
        n_visits = np.repeat(10, self.actions.size)
        return {str(a.value): a for a in self.surrogate_actions}, None, np.c_[self.actions, weighted_q_values, n_visits]

    def compute_expected_value_from_offer(self, action, observation, opponent_model, iteration_number):
        # Compute trial reward
        response, _, probabilities = opponent_model.forward(observation, action)
        reward = self.utility_function(action.value, response) @ probabilities
        return reward


class DoMZeroSender(DoMZeroModel):

    def __init__(self,
                 actions,
                 softmax_temp: float,
                 threshold: Optional[float],
                 prior_belief: np.array,
                 opponent_model: SubIntentionalAgent,
                 seed: int):
        super().__init__(actions, softmax_temp, threshold, prior_belief, opponent_model, seed)
        self.config = get_config()
        self.belief = DoMZeroSenderBelief(prior_belief[:, 0], prior_belief[:, 1], self.opponent_model, self.history)
        self.environment_model = DoMZeroSenderEnvironmentModel(self.opponent_model, self.utility_function,
                                                               actions, self.belief)
        self.solver = DoMZeroSenderSolver(self.potential_actions, self.belief, self.opponent_model,
                                          self.utility_function,
                                          float(self.config.get_from_env("planning_depth")),
                                          float(self.config.get_from_env("discount_factor")))
        self.name = "DoM(0)_sender"
        self.alpha = 0.0

    def utility_function(self, action, observation, *args, **kwargs):
        """

        :param action: bool - either True for accepting the offer or False for rejecting it
        :param observation: float - representing the current offer
        :return:
        """
        if len(args) == 0:
            receiver_counter_action = observation
        else:
            receiver_counter_action = args[0]
        game_reward = (1 - action - self.threshold) * receiver_counter_action
        # self.history.rewards.append(game_reward)
        return game_reward

    def update_nested_models(self, action=None, observation=None, iteration_number=None):
        pass

    def post_action_selection_update_nested_models(self, action=None, iteration_number=None):
        pass

    @staticmethod
    def get_aleph_mechanism_status(sequence=False):
        # Senders have no paranoia detection - Aleph is Receiver-side only
        if sequence:
            return [False]
        return False
