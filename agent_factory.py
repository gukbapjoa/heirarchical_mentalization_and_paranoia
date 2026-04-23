from agents_models.intentional_agents.tom_two_agents.tom_two_agents import *


class AgentFactory:
    def __init__(self):
        self.config = get_config()
        self.device = self.config.device
        # Load environmental parameters
        self.softmax_temp = float(self.config.softmax_temperature)
        self.exploration_bonus = float(self.config.get_from_env("uct_exploration_bonus"))
        self.subintentional_weight = float(self.config.get_from_env("subintentional_weight"))
        self.agent_actions = np.round(np.arange(0, 1.05, float(self.config.get_from_general("offers_step_size"))), 2)
        self.subject_actions = np.array([True, False])
        self.include_random = bool(self.config.get_from_general("include_random"))
        self.task_duration = self.config.get_from_env("n_trials")
        self.sender_theta = list(self.config.get_from_general("sender_thresholds"))
        self._sender_theta = self.sender_theta if self.config.get_from_general("number_of_rational_agents") > 1 else self.sender_theta[:-1]
        self.sender_theta = self._sender_theta if self.include_random else self._sender_theta[1:]
        self.receiver_theta = list(self.config.get_from_general("receiver_thresholds"))
        self.grid_size = 0
        self.include_subject_threshold = self.config.get_from_env("subintentional_type")
        self.path_to_memoization_data = self.config.path_to_memoization_data
        self.aleph_ipomdp_delta_parameter = float(self.config.get_from_general("strong_typicality_delta"))
        self.aleph_ipomdp_omega_parameter = float(self.config.get_from_general("expected_reward_omega"))
        self.aleph_ipomdp_awareness = bool(self.config.get_from_general("aleph_ipomdp_awareness"))

    def create_experiment_grid(self):
        receiver_parameters = self.receiver_theta
        receiver_grid_size = len(receiver_parameters)
        sender_parameters = self.sender_theta
        sender_grid_size = len(sender_parameters)
        self.grid_size = sender_grid_size * receiver_grid_size
        return {"sender_parameters": sender_parameters,
                "receiver_parameters": receiver_parameters}

    @staticmethod
    def _create_prior_distribution(support):
        thresholds_probabilities = np.repeat(1 / len(support), len(support))
        return np.array([support, thresholds_probabilities]).T

    def constructor(self, agent_role: str, agent_dom_level: str = None):
        agent = None
        if agent_dom_level is None:
            agent_dom_level = self.config.get_agent_tom_level(agent_role)
        if agent_dom_level == "DoM-1":
            agent = self.dom_minus_one_constructor(agent_role)
        if agent_dom_level == "DoM0":
            agent = self.dom_zero_constructor(agent_role)
        if agent_dom_level == "DoM1":
            agent = self.dom_one_constructor(agent_role)
        if agent_dom_level == "DoM2":
            agent = self.dom_two_constructor(agent_role)
        return agent

    # Coding for DoM(-1) Sender or DoM(-1) Receiver
    def dom_minus_one_constructor(self, agent_role):
        if agent_role == "rational_sender":
            if self.config.subintentional_agent_type == "uniform":
                agent = UniformRationalRandomSubIntentionalSender(self.agent_actions, self.softmax_temp,
                                                                  self.subintentional_weight)
            else:
                agent = SoftMaxRationalRandomSubIntentionalSender(self.agent_actions, self.softmax_temp,
                                                                  self.subintentional_weight)
        else:
            agent = SubIntentionalReceiver(self.subject_actions, self.softmax_temp)
        return agent

    # Coding for DoM(0) Sender  or Dom(0) Receiver
    def dom_zero_constructor(self, agent_role, aleph_ipomdp_activated: bool = True):
        if agent_role == "rational_sender":
            opponent_model = self.dom_minus_one_constructor("rational_receiver")
            opponent_theta_hat_distribution = self._create_prior_distribution(self.sender_theta)
            output_agent = DoMZeroSender(self.agent_actions, self.config.softmax_temperature, 0.0,
                                         opponent_theta_hat_distribution, opponent_model, self.config.seed)
        else:
            opponent_model = self.dom_minus_one_constructor("rational_sender")
            opponent_theta_hat_distribution = self._create_prior_distribution(self.sender_theta)
            output_agent = DoMZeroReceiver(self.subject_actions, self.config.softmax_temperature, 0.0,
                                           opponent_theta_hat_distribution, opponent_model, self.config.seed,
                                           self.task_duration,
                                           aleph_ipomdp_activated,
                                           self.aleph_ipomdp_delta_parameter,
                                           self.aleph_ipomdp_omega_parameter)
        return output_agent

    # Coding for DoM(1) Sender
    def dom_one_constructor(self, agent_role, nested=False):
        if agent_role == "rational_sender":
            opponent_model = self.dom_zero_constructor("rational_receiver")
            opponent_theta_hat_distribution = self._create_prior_distribution(self.receiver_theta)
            memoization_table = DoMOneMemoization(self.path_to_memoization_data)
            output_agent = DoMOneSender(self.agent_actions, self.config.softmax_temperature, None,
                                        memoization_table, opponent_theta_hat_distribution, opponent_model,
                                        self.config.seed, nested)
            
        else:
        # NEW: DoMOneReceiver wraps a DoMZeroSender opponent
            opponent_model = self.dom_zero_constructor("rational_sender")
            opponent_theta_hat_distribution = self._create_prior_distribution(self.sender_theta)
            memoization_table = DoMOneMemoization(self.path_to_memoization_data)
            output_agent = DoMOneReceiver(self.subject_actions, self.config.softmax_temperature, None,
                                      memoization_table, opponent_theta_hat_distribution, opponent_model,
                                      self.config.seed, int(self.task_duration), nested)    
            
        return output_agent
    

    # Coding for DoM(2) Receiver
    def dom_two_constructor(self, agent_role):
        if agent_role == "rational_sender":
        # NEW: DoMTwoSender wraps a DoMOneReceiver opponent
            opponent_model = self.dom_one_constructor("rational_receiver", True)
            opponent_theta_hat_distribution = self._create_prior_distribution(self.receiver_theta)
            memoization_table = DoMTwoMemoization(self.path_to_memoization_data)
            output_agent = DoMTwoSender(self.agent_actions, self.config.softmax_temperature, None,
                                    memoization_table, opponent_theta_hat_distribution,
                                    opponent_model, self.config.seed)
        
        else:
        # Existing: DoMTwoReceiver wraps a DoMOneSender opponent
            opponent_model = self.dom_one_constructor("rational_sender", True)
            opponent_theta_hat_distribution = self._create_prior_distribution(self.sender_theta)
            memoization_table = DoMTwoMemoization(self.path_to_memoization_data)
            output_agent = DoMTwoReceiver(self.subject_actions, self.config.softmax_temperature, None,
                                      memoization_table, opponent_theta_hat_distribution,
                                      opponent_model, self.config.seed, self.task_duration)
        
        return output_agent

