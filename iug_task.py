from typing import Tuple
from agents_models.abstract_agents import *


class IUG:

    def __init__(self, seed):
        self.config = get_config()
        self.n_trails = int(self.config.get_from_env("n_trials"))
        self.endowment = float(self.config.get_from_env("endowment"))
        self.seed = seed
        self.trail_results = []

    def reset(self):
        self.trail_results = []

    def export_nested_beliefs(self, beliefs: Optional[Union[dict, np.ndarray]], 
                          supports: Optional[Union[dict, np.ndarray]], agent_name: str,
                          agent_dom_level, receiver_threshold: str, agent_threshold: str,
                          recursion_depth: int):
        # Guard: if beliefs is a plain array (innermost level), export directly
        if not isinstance(beliefs, dict):
            prefix = 'p' + f'_{agent_dom_level}_{recursion_depth}'
            return self.export_type_beliefs(beliefs, prefix, supports, agent_name,
                                            receiver_threshold, agent_threshold)
        nested_beliefs = dict()
        for level in beliefs.keys():
            if level == 'zero_order_belief':
                prefix = 'p' + f'_{agent_dom_level}_{recursion_depth}'
                belief_df = self.export_type_beliefs(beliefs[level], prefix, supports[level], agent_name,
                                                     receiver_threshold, agent_threshold)
            elif level == 'nested_beliefs' and recursion_depth < 2:
                prefix = 'q' + f'_{agent_dom_level}_{recursion_depth}'
                belief_df = self.export_type_beliefs(beliefs[level], prefix, supports[level], agent_name,
                                                     receiver_threshold, agent_threshold)
            else:
                belief_df = self.export_nested_beliefs(beliefs[level], supports[level], agent_name, agent_dom_level,
                                                       receiver_threshold, agent_threshold, recursion_depth - 1)
            nested_beliefs[level] = belief_df
        unified_df = nested_beliefs['zero_order_belief'].merge(nested_beliefs['nested_beliefs'])
        unified_df = unified_df.iloc[-self.n_trails:].reset_index(drop=True)
        return unified_df

    def export_type_beliefs(self, beliefs: Optional[np.array], columns_prefix: str,
                            support, agent_name: str,
                            receiver_threshold: str, agent_threshold: str):
        beliefs_df = None
        if not pd.isna(beliefs).all():
            columns = [columns_prefix + "_" + str(x) for x in np.arange(len(support))]
            beliefs_df = pd.DataFrame(beliefs, columns=columns)
            beliefs_df['agent_name'] = agent_name
            beliefs_df['seed'] = self.seed
            beliefs_df['trial_number'] = np.arange(0, beliefs_df.shape[0], 1)
            beliefs_df = self.add_experiment_data_to_df(beliefs_df, receiver_threshold, agent_threshold)
        return beliefs_df

    @staticmethod
    def add_experiment_data_to_df(df: pd.DataFrame, receiver_threshold: str, sender_threshold: str) -> pd.DataFrame:
        df['receiver_threshold'] = receiver_threshold
        df['sender_threshold'] = sender_threshold
        return df

    def simulate_task(self, sender, receiver, receiver_threshold: str, sender_threshold: str) -> \
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        seed = self.seed
        q_values_list = []
        offer = Action(None, False)
        response = Action(None, False)
        for trial_number in range(0, self.n_trails, 1):
            print(f'Starting trial number {trial_number}', flush=True)
            offer, response, trial_results, q_values = self.trial(trial_number, sender, receiver, seed, offer, response)
            self.trail_results.append(trial_results)
            q_values_list.append(q_values)
            print(f'Ending trial number {trial_number}', flush=True)
        experiment_results = pd.DataFrame(self.trail_results, columns=['offer', 'offer_probability',
                                                                       'response', 'response_probability',
                                                                       'sender_reward', 'receiver_reward'])
        experiment_results['trial_number'] = np.arange(1, self.n_trails+1, 1)
        experiment_results['seed'] = self.seed
        experiment_results = self.add_experiment_data_to_df(experiment_results, receiver_threshold,
                                                            sender_threshold)
        agents_q_values = pd.concat(q_values_list)
        agents_q_values.columns = ['action', 'q_value', 'agent', 'parameter', 'trial_number']
        agents_q_values['seed'] = self.seed
        agents_q_values = self.add_experiment_data_to_df(agents_q_values, receiver_threshold,
                                                         sender_threshold)
        # RECEIVER BELIEF EXPORT
        if receiver.name == 'DoM(0)_receiver':
            receiver_belief = self.export_type_beliefs(receiver.belief.belief_distribution, "p_0",
                                                    receiver.belief.support,
                                                    receiver.name, receiver_threshold, sender_threshold)
        elif receiver.name == 'DoM(1)_receiver':
            receiver_belief = self.export_nested_beliefs(receiver.belief.belief_distribution,
                                                        receiver.belief.supports,
                                                        receiver.name, "1", receiver_threshold, sender_threshold,
                                                        1)
        else:
            receiver_belief = self.export_nested_beliefs(receiver.belief.belief_distribution,
                                                        receiver.belief.supports,
                                                        receiver.name, "2", receiver_threshold, sender_threshold,
                                                        2)
        
        # SENDER BELIEF EXPORT
        if sender.name == 'DoM(1)_sender':
            sender_belief = self.export_nested_beliefs(sender.belief.belief_distribution,
                                                    sender.belief.supports,
                                                    sender.name, "1", receiver_threshold, sender_threshold,
                                                    1)
        elif sender.name == 'DoM(2)_sender':
            sender_belief = self.export_nested_beliefs(sender.belief.belief_distribution,
                                                    sender.belief.supports,
                                                    sender.name, "2", receiver_threshold, sender_threshold,
                                                    2)
        else:
            sender_belief = self.export_type_beliefs(sender.belief.belief_distribution, "p(-1)",
                                                    sender.belief.support,
                                                    sender.name, receiver_threshold, sender_threshold)
        if self.config.env == "aleph_ipomdp":
            mental_states = receiver.get_aleph_mechanism_status(True)
            if isinstance(mental_states, bool):
                mental_states = [mental_states] * self.n_trails
            receiver_mental_state = pd.DataFrame({'mental_state': mental_states})
            receiver_mental_state['trial_number'] = np.arange(0, receiver_mental_state.shape[0], 1)
            receiver_mental_state['seed'] = self.seed
            receiver_mental_state['sender_threshold'] = sender.threshold
        
        else:
             receiver_mental_state = None
        return experiment_results, agents_q_values, receiver_belief, sender_belief, receiver_mental_state

    @staticmethod
    def trial(trial_number, sender,  receiver, seed, offer, response):
        offer, offer_probability, sender_q_values, sender_policy = sender.act(seed, offer, response, trial_number)
        response, response_probability, receiver_q_values, receiver_policy = receiver.act(seed, response, offer, trial_number + 1)
        sender_reward = (1-offer.value) * response.value
        receiver_reward = offer.value * response.value
        sender.history.rewards.append(sender_reward)
        receiver.history.rewards.append(receiver_reward)
        sender_q_values = pd.DataFrame(sender_q_values)
        sender_q_values['agent_name'] = sender.name
        sender_q_values['parameter'] = sender.threshold
        sender_q_values['trial'] = trial_number
        receiver_q_values = pd.DataFrame(receiver_q_values)
        receiver_q_values['agent_name'] = receiver.name
        receiver_q_values['parameter'] = receiver.threshold
        receiver_q_values['trial'] = trial_number
        q_values = pd.concat([sender_q_values, receiver_q_values])
        return offer, response, np.array([offer.value, offer_probability,
                                          response.value, response_probability,
                                          sender_reward, receiver_reward]), q_values

