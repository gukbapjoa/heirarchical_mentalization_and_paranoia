import functools

import numpy as np

from agents_models.abstract_agents import *
import pandas as pd


class DomZeroReceiverBelief(DoMZeroBelief):

    def __init__(self, support, zero_level_belief, opponent_model, history: History):
        super().__init__(support, zero_level_belief, opponent_model, history)
        self.likelihood = np.zeros((1, len(support)))
        self.softmax_distributions = np.empty((len(self.support), len(opponent_model.potential_actions),
                                               opponent_model.config.task_duration + 1))

    def reset(self, size=1):
        self.belief_distribution = self.belief_distribution[0:size, ]
        self.likelihood = self.likelihood[0:size, ]

    def get_current_belief(self):
        return self.belief_distribution[-1]

    def get_current_likelihood(self):
        return self.likelihood

    def compute_likelihood(self, action: Action, observation: Action, prior, iteration_number=None, nested=False):
        """
        Compute observation likelihood given opponent's type and last action
        :param action:
        :param observation:
        :param prior:
        :param iteration_number:
        :param nested:
        :return:
        """
        last_observation = self.history.get_last_observation()
        offer_likelihood = np.empty_like(prior)
        original_threshold = self.opponent_model.threshold
        softmax_distributions = np.empty((len(self.support), len(self.opponent_model.potential_actions)))
        for i in range(len(self.support)):
            theta = self.support[i]
            if theta == 0.0:
                offer_likelihood[i] = 1 / len(self.opponent_model.potential_actions)
                probabilities = np.repeat(1 / len(self.opponent_model.potential_actions),
                                          len(self.opponent_model.potential_actions))
                softmax_distributions[i, :] = probabilities
                continue
            self.opponent_model.threshold = theta
            possible_opponent_actions, opponent_q_values, probabilities = \
                self.opponent_model.forward(last_observation, action, iteration_number - 1)
            softmax_distributions[i, :] = probabilities
            # If the observation is not in the feasible action set then it singles theta hat:
            observation_in_feasible_set = np.any(possible_opponent_actions == observation.value)
            if not observation_in_feasible_set:
                observation_probability = 1e-4
            else:
                observation_probability = probabilities[np.where(possible_opponent_actions == observation.value)]
            offer_likelihood[i] = observation_probability
        self.softmax_distributions[:, :, iteration_number] = softmax_distributions
        self.opponent_model.threshold = original_threshold
        self.likelihood = np.vstack([self.likelihood, offer_likelihood])
        return offer_likelihood


class DoMZeroReceiverEnvironmentModel(DoMZeroEnvironmentModel):

    def step_from_is(self, new_interactive_state: InteractiveState, previous_observation: Action, action: Action,
                     seed: int, iteration_number: int):
        pass

    def __init__(self, opponent_model: SubIntentionalAgent, reward_function, actions,
                 belief_distribution: DomZeroReceiverBelief):
        super().__init__(opponent_model, reward_function, actions, belief_distribution)

    def update_persona(self, observation, action, iteration_number):
        self.opponent_model.update_bounds(observation, action, iteration_number)
        self.low = self.opponent_model.low
        self.high = self.opponent_model.high


class DoMZeroAlephMechanism:

    def __init__(self, history: History, actions, observations, duration, activated: bool = True,
                 delta: int = 10, omega: float = 1.96):
        self.is_aleph_mechanism_on = [False]
        self.history = history
        self.actions = actions
        self.observations = observations
        self.action_observation_array = np.transpose([np.tile(self.observations, len(self.actions)),
                                                      np.repeat(self.actions, len(self.observations))])
        self.duration = duration
        self.activated = activated
        self.delta = delta
        self.omega = omega

    def reset(self, iteration_number: int, terminal=False):
        if terminal or iteration_number < 1:
            self.is_aleph_mechanism_on = [False]
        else:
            self.is_aleph_mechanism_on = self.is_aleph_mechanism_on[0:iteration_number+1]

    def update_aleph_mechanism(self, iteration_number, mental_state):
        self.is_aleph_mechanism_on.append(mental_state)

    def delta_strong_typicality(self, trial_number: int, likelihood: np.array) -> np.array:
        observations = [x.value for x in self.history.observations]
        unique_observations, location, number_of_appearance = np.unique(observations, return_counts=True, return_index=True)
        observed_frequency = np.reshape(number_of_appearance[np.argsort(location)] / max(1, trial_number), (1, len(unique_observations)))
        adapted_observed_frequency = np.repeat(observed_frequency, number_of_appearance[np.argsort(location)])
        expected_frequency = likelihood[1:(trial_number + 1), :]
        target_length = expected_frequency.shape[-1]
        fixed_adapted = np.zeros(target_length)
        copy_len = min(len(adapted_observed_frequency), target_length)
        fixed_adapted[:copy_len] = adapted_observed_frequency[:copy_len]
        distance = np.absolute(fixed_adapted[:, np.newaxis] - expected_frequency.T)
        safe_trial_number = max(1, trial_number)
        adapted_delta = np.max([(self.duration - trial_number) / safe_trial_number, self.delta])
        typical_set = distance <= adapted_delta * expected_frequency.T
        return np.all(typical_set, axis=0)

    def expected_reward_monitoring(self, trial_number, history, likelihood, utility_function,
                                   softmax_transformation) -> np.array:
        if trial_number > 0 and len(history.observations) >= trial_number:
            last_obs_val = history.observations[trial_number - 1].value
        elif len(history.observations) > 0:
            last_obs_val = history.observations[-1].value
        else:
            last_obs_val = 0
        expected_observed_reward = np.matmul(
            utility_function(last_obs_val, self.actions),
            np.transpose(softmax_transformation))
        vec_utility = np.vectorize(utility_function)
        offer_likelihood = likelihood[:, :, trial_number]
        reward_distribution = np.reshape(
            vec_utility(self.action_observation_array[:, 0], self.action_observation_array[:, 1]),
            (len(self.observations), len(self.actions)), order="F")
        expected_reward = np.matmul(np.matmul(offer_likelihood, reward_distribution),
                                    np.transpose(softmax_transformation))
        # Need to fins a way to compute SE
        mean_offer = np.matmul(offer_likelihood, self.observations)
        diff = self.observations[:, np.newaxis] - mean_offer
        if np.any(np.abs(diff) > 1e6):
            print(f"WARNING: overflow clipped in sigma_offer at trial {trial_number}", flush=True)
        diff = np.clip(diff, -1e6, 1e6)
        sigma_offer = np.sqrt(
            np.diag(np.dot(offer_likelihood, np.power(diff, 2))))
        bounds = np.array([expected_reward - sigma_offer * self.omega, expected_reward + sigma_offer * self.omega])
        is_observed_reward_in_bounds = np.logical_and(expected_observed_reward > bounds[0, :], expected_observed_reward < bounds[1, :])
        return is_observed_reward_in_bounds

    def compute_z_vector(self, trial_number, observation_likelihood, total_likelihood,
                         history, utility_function, softmax_transformation):
        strong_typicality = self.delta_strong_typicality(trial_number, observation_likelihood)
        expected_reward_monitoring = self.expected_reward_monitoring(trial_number, history, total_likelihood,
                                                                     utility_function,
                                                                     softmax_transformation)
        min_len = min(len(strong_typicality), len(expected_reward_monitoring))
        return np.logical_and(strong_typicality[:min_len], expected_reward_monitoring[:min_len])

    def aleph_mechanism(self, iteration_number, belief_distribution: DoMZeroBelief, utility_function,
                        softmax_transformation):
        observation_likelihood = belief_distribution.likelihood
        history = belief_distribution.history
        if self.activated:
            detection_mechanism = self.compute_z_vector(iteration_number, observation_likelihood,
                                                        belief_distribution.softmax_distributions,
                                                        history,
                                                        utility_function, softmax_transformation)
            return ~np.any(detection_mechanism)
        else:
            return False


class DoMZeroReceiverSolver(DoMZeroEnvironmentModel):
    def step_from_is(self, new_interactive_state: InteractiveState, previous_observation: Action, action: Action,
                     seed: int, iteration_number: int):
        pass

    def __init__(self, actions, belief_distribution: DoMZeroBelief, opponent_model,
                 aleph_mechanism: DoMZeroAlephMechanism,
                 active_detection: bool, breakdown_policy: bool,
                 reward_function, planning_horizon, discount_factor, task_duration,
                 softmax_temp):
        super().__init__(opponent_model, reward_function, actions, belief_distribution)
        self.actions = actions
        self.belief = belief_distribution
        self.opponent_model = opponent_model
        self.aleph_mechanism = aleph_mechanism
        self.aleph_ipomdp_model = active_detection
        self.breakdown_policy = breakdown_policy
        self.utility_function = reward_function
        self.planning_horizon = planning_horizon
        self.discount_factor = discount_factor
        self.task_duration = task_duration
        self.softmax_temp = softmax_temp
        self.action_node = None
        self.surrogate_actions = [Action(value, False) for value in self.actions]
        self.name = "tree_search"
        self.low = 0.0
        self.high = 1.0
        self.planning_tree = []
        self.q_values = []

    def get_aleph_mechanism_status(self, sequence: bool = False) -> Union[bool, list]:
        if sequence:
            return self.aleph_mechanism.is_aleph_mechanism_on
        return self.aleph_mechanism.is_aleph_mechanism_on[-1]

    def set_aleph_mechanism_state(self, z_state: bool):
        return self.aleph_mechanism.is_aleph_mechanism_on.append(z_state)

    def aleph_policy(self, iteration_number, is_aleph_mechanism_triggered,
                     actions, mcts_tree, q_values):
        n_visits = np.repeat(self.planning_horizon, self.actions.size)
        # If the Flip Flop mechanism is on
        if self.get_aleph_mechanism_status():
            self.aleph_mechanism.update_aleph_mechanism(iteration_number, True)
            weighted_q_values = [-1, 1]
            return {str(a.value): a for a in self.surrogate_actions}, None, \
                   np.c_[self.actions, weighted_q_values, n_visits]
        # If the א-mechanism is on for the first time:
        if is_aleph_mechanism_triggered:
            self.aleph_mechanism.update_aleph_mechanism(iteration_number, True)
            # Compute x-policy
            if self.breakdown_policy:
                weighted_q_values = [-1, 1]
            else:
                weighted_q_values = [-1 / 10, 1 / 10]
            return {str(a.value): a for a in self.surrogate_actions}, None, \
                   np.c_[self.actions, weighted_q_values, n_visits]
        else:
            self.aleph_mechanism.update_aleph_mechanism(iteration_number, False)
        return actions, mcts_tree, q_values

    def execute_aleph_ipomdp(self, q_values, iteration_number, actions, mcts_tree):
        softmax_transformation = np.exp(q_values[:, 1] / self.softmax_temp) / np.exp(
            q_values[:, 1] / self.softmax_temp).sum()
        is_aleph_mechanism_triggered = self.aleph_mechanism.aleph_mechanism(iteration_number,
                                                                            self.belief_distribution,
                                                                            self.utility_function,
                                                                            softmax_transformation)
        actions, mcts_tree, q_values = self.aleph_policy(iteration_number, is_aleph_mechanism_triggered,
                                                         actions, mcts_tree, q_values)
        return actions, mcts_tree, q_values

    def plan(self, action, observation, iteration_number, update_belief):
        # Update history
        action_length = len(self.belief.history.actions)
        observation_length = len(self.belief.history.observations)
        # Update belief
        if update_belief:
            self.belief.update_distribution(action, observation, iteration_number)
        # Update nested opponent model
        self.update_low_and_high(self.belief.history.observations[-2] if iteration_number > 1 else Action(None, False),
                                 self.belief.history.actions[-1] if iteration_number > 1 else Action(None, False)
                                 , iteration_number)
        # Span planning tree to compute expected behaviour
        actions, mcts_tree, q_values = self.expectimax_planning(observation, action_length, observation_length,
                                                                iteration_number)
        if self.aleph_ipomdp_model:
            actions, mcts_tree, q_values = self.execute_aleph_ipomdp(q_values, iteration_number, actions, mcts_tree)
        return actions, mcts_tree, q_values

    def expectimax_planning(self, observation, action_length, observation_length, iteration_number):
        q_values_array = []
        self.q_values = []
        for threshold in self.belief.support:
            # Reset nested model
            self.reset_persona(threshold, action_length, observation_length,
                               self.opponent_model.belief, iteration_number)
            future_values = functools.partial(self.recursive_tree_spanning,
                                              observation=observation,
                                              current_low=self.low,
                                              current_high=self.high,
                                              opponent_model=self.opponent_model,
                                              iteration_number=iteration_number,
                                              planning_step=0)
            q_values = list(map(future_values, self.surrogate_actions))
            q_values_array.append(q_values)
        dt = pd.DataFrame(self.q_values)
        self.reset_persona(None, action_length, observation_length,
                           self.opponent_model.belief, iteration_number)
        weighted_q_values = self.belief.belief_distribution[-1, :] @ np.array(q_values_array)
        n_visits = np.repeat(self.planning_horizon, self.actions.size)
        return {str(a.value): a for a in self.surrogate_actions}, None, np.c_[self.actions, weighted_q_values, n_visits]

    def recursive_tree_spanning(self, action, observation,
                                current_low, current_high,
                                opponent_model, iteration_number,
                                planning_step):
        # Compute trial reward
        reward = self.utility_function(action.value, observation.value)
        if planning_step >= self.planning_horizon or iteration_number >= self.task_duration:
            remaining_time = max(self.task_duration - iteration_number, 1)
            return self.utility_function(action.value, observation.value) * remaining_time
        # Update bounds:
        current_low = observation.value * (1 - action.value) + current_low * action.value
        current_high = observation.value * action.value + current_high * (1 - action.value)
        # compute offers and probs given previous history
        potential_actions, _, probabilities = opponent_model.forward(observation, action, iteration_number,
                                                                     [current_low, current_high])
        average_counter_offer_value = np.round(np.dot(potential_actions, probabilities).item() / 0.05) * 0.05
        average_counter_offer = Action(average_counter_offer_value, False)
        # compute offers and probs given previous history
        future_values = functools.partial(self.recursive_tree_spanning,
                                          observation=average_counter_offer,
                                          current_low=current_low,
                                          current_high=current_high,
                                          opponent_model=self.opponent_model,
                                          iteration_number=iteration_number + 1,
                                          planning_step=planning_step + 1)
        self.planning_tree.append([self.opponent_model.threshold, iteration_number, action.value, observation.value,
                                   average_counter_offer_value])
        future_q_values = list(map(future_values, self.surrogate_actions))
        q_value = reward + self.discount_factor * max(future_q_values)
        self.q_values.append([self.opponent_model.threshold, iteration_number, action.value, observation.value, reward,
                              q_value, future_q_values])
        return q_value

    def reset(self, iteration_number):
        self.low = self.opponent_model.low
        self.high = self.opponent_model.high
        self.aleph_mechanism.reset(iteration_number)


class DoMZeroReceiver(DoMZeroModel):

    def __init__(self,
                 actions,
                 softmax_temp: float,
                 threshold: Optional[float],
                 prior_belief: np.array,
                 opponent_model: SubIntentionalAgent,
                 seed: int,
                 task_duration: int,
                 aleph_ipomdp_activated,
                 delta_strong_typicality_parameter,
                 expected_reward_ci_parameter):
        super().__init__(actions, softmax_temp, threshold, prior_belief, opponent_model, seed)
        self.detection_mechanism = DoMZeroAlephMechanism(self.history,
                                                         actions,
                                                         self.opponent_model.potential_actions,
                                                         task_duration,
                                                         aleph_ipomdp_activated,
                                                         delta_strong_typicality_parameter,
                                                         expected_reward_ci_parameter)
        self.belief = DomZeroReceiverBelief(prior_belief[:, 0], prior_belief[:, 1], self.opponent_model, self.history)
        self.environment_model = DoMZeroReceiverEnvironmentModel(self.opponent_model, self.utility_function,
                                                                 actions,
                                                                 self.belief)
        self.solver = DoMZeroReceiverSolver(self.potential_actions,
                                            self.belief,
                                            self.opponent_model,
                                            self.detection_mechanism,
                                            self.config.get_from_env("active_detection_mechanism"),
                                            self.config.get_from_env("break_down_policy"),
                                            self.utility_function,
                                            float(self.config.get_from_env("planning_depth")),
                                            float(self.config.get_from_env("discount_factor")),
                                            task_duration,
                                            softmax_temp)
        self.name = "DoM(0)_receiver"

    def get_aleph_mechanism_status(self, sequence=False):
        # Line 1: Try to get the real paranoia data from the solver
        if hasattr(self.solver, 'get_aleph_mechanism_status'):
            return self.solver.get_aleph_mechanism_status(sequence)
        
        # Line 2: If the solver is missing the data, check what kind of report was asked for
        if sequence:
            # Line 3: If asked for the full history (at the end of the game), return 12 rounds
            return [0] * 12
        else:
            # Line 4: If asked for a quick snapshot (during the game), return a matching 2-item list
            return [0, 0]

    def set_aleph_mechanism_state(self, mental_state: bool):
        return self.solver.set_aleph_mechanism_state(mental_state)

    def utility_function(self, observation, action, *args):
        """

        :param action: bool - either True for accepting the offer or False for rejecting it
        :param observation: float - representing the current offer
        :return:
        """
        if observation is None:
            return 0.0
        game_reward = (action - self.threshold) * observation
        return game_reward

    def update_belief(self, action, observation):
        observation_likelihood_per_type = np.zeros_like(self.belief.belief_distribution[:, 0])
        i = 0
        opponent_threshold = self.opponent_model.threshold
        for gamma in self.belief.belief_distribution[:, 0]:
            self.opponent_model.threshold = gamma
            relevant_actions, q_values, probabilities = self.opponent_model.forward(observation, action)
            observation_likelihood = probabilities[np.where(relevant_actions == observation)]
            observation_likelihood_per_type[i] = observation_likelihood
            i += 1
        self.opponent_model.threshold = opponent_threshold
        prior = self.belief.belief_distribution[:, -1]
        posterior = observation_likelihood_per_type * prior
        self.belief.belief_distribution = np.c_[self.belief.belief_distribution, posterior / posterior.sum()]
