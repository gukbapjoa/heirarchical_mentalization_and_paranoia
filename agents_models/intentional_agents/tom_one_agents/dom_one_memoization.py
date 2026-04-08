from pandas.errors import DataError
from Solver.utils.memoization_table import *
import os
from os.path import exists


belief_columns = ["p_1_0", "p_1_1", "q0", "q1", "q2", "trial_number", "seed", "sender_threshold"]
history_columns = ["offer", "response", "trial_number", "seed", "sender_threshold"]
q_values_columns = ["action", "q_value", "trial_number", "seed", "sender_threshold"]


class DoMOneMemoization(MemoizationTable):

    def __init__(self, path_to_memoization_dir, softmax_temp=0.1):
        self.softmax_temp = softmax_temp
        self.config = get_config()
        self.default_columns = self._get_memoization_columns()
        self.columns = self.default_columns
        self._table_name = f'DoM_1_unified_memoization_data_softmax_temp_{self.softmax_temp}'
        self.target_table_name = f'{self._table_name}.csv'
        self.path_to_dir = path_to_memoization_dir
        self.path_to_table = os.path.join(self.path_to_dir, self.target_table_name)
        self.use_memoization = bool(self.config.get_from_general("use_memoization"))
        super().__init__(path_to_memoization_dir)
        self.path_to_buffer_file = self.create_buffer_file()

    def create_buffer_file(self):
        if not os.path.exists(self.path_to_dir):
            os.mkdir(self.path_to_dir)
        buffer_file_name = f'{self._table_name}_buffer.csv'
        return os.path.join(self.path_to_dir, buffer_file_name)

    def create_unified_memoization_table(self):
        q_values = self._read_and_process_table("q_values")
        game_results = self._read_and_process_table("simulation_results")
        nested_beliefs = self._read_and_process_table("beliefs")
        data = self.combine_results(q_values, game_results, nested_beliefs)
        self.columns = data.columns

    def load_data(self):
        # First - see if we already have memoization data there
        if self.use_memoization:
            print(self.path_to_dir, flush=True)
            if len(os.listdir(self.path_to_dir)) > 0:
                print('Load memoization data', flush=True)
                data = []
                files = os.listdir(self.path_to_dir)
                for file in files:
                    df = pd.read_csv(f'{self.path_to_dir}/{file}', header=None)
                    data.append(df)
                df = pd.concat(data, axis=0, ignore_index=True)
                df.columns = self.columns
                return df
            # If not - we create the data
            else:
                try:
                    print('Loading from memory', flush=True)
                    q_values = self._read_and_process_table("q_values")
                    game_results = self._read_and_process_table("simulation_results")
                    nested_beliefs = self._read_and_process_table("beliefs")
                    data = self.combine_results(q_values, game_results, nested_beliefs)
                except FileNotFoundError:
                    print('First time simulating - no data!')
                    data = None
        else:
            data = None
        return data

    def save_data(self):
        if not os.path.exists(self.path_to_dir):
            os.mkdir(self.path_to_dir)
        self.data.to_csv(self.path_to_table, index=False)

    def update_buffer_data(self, new_data):
        # If the buffer file already exists
        if exists(self.path_to_buffer_file):
            new_data.to_csv(self.path_to_buffer_file, mode='a', index=False, header=False)
        else:
            new_data.to_csv(self.path_to_buffer_file, mode='w', index=False, header=False)

    def _read_and_process_table(self, directory_name):
        """
        Method to load tables from memory
        :param directory_name:
        :return:
        """
        data = []
        path = f'{os.path.dirname(self.config.planning_results_dir)}/{directory_name}'
        if directory_name == "beliefs":
            path = f'{self.config.beliefs_dir}/sender_beliefs'
        files = os.listdir(path)
        if len(files) == 0:
            raise FileNotFoundError('No files founds')
        for file in files:
            df = pd.read_csv(f'{path}/{file}')
            data.append(df)
        return pd.concat(data, axis=0, ignore_index=True)

    @staticmethod
    def combine_results(raw_q_values: pd.DataFrame, raw_game_results: pd.DataFrame, raw_beliefs: pd.DataFrame):
        """
        Method to merge Q values, history and beliefs into a single view
        :param raw_q_values:
        :param raw_game_results:
        :param raw_beliefs:
        :return:
        """
        # Filter Q-values
        q_values = raw_q_values.loc[raw_q_values['agent'] == 'DoM(1)_sender'][q_values_columns]
        # Compose game history
        history = raw_game_results.loc[raw_game_results['sender_threshold'] > 0][history_columns]
        # filter nested beliefs
        beliefs = raw_beliefs.loc[raw_beliefs['agent_name'] == 'DoM(1)_sender'][belief_columns]
        # round beliefs
        beliefs = beliefs.assign(q1=np.round(beliefs["0.0"], 3),
                                 q2=np.round(beliefs["0.3"], 3),
                                 q3=np.round(beliefs["0.7"], 3))
        beliefs['trial_number'] = beliefs['trial_number']+1
        # Join tables to get unified view
        q_values_and_beliefs = pd.merge(q_values, beliefs, on=["trial_number", "seed", "sender_threshold"])
        q_values_and_beliefs_and_history = pd.merge(q_values_and_beliefs, history, on=["trial_number", "seed",
                                                                                       "sender_threshold"])
        return q_values_and_beliefs_and_history

    def query_table(self, query_parameters: dict):
        """
        Q values memoization: takes as input game settings and returns mean(q-values) if not empty
        :param query_parameters:
        :return: pd.dataframe
        """
        if self.data is None:
            q_values = pd.DataFrame()
            return q_values
        trial = query_parameters['trial']
        threshold = query_parameters['threshold']
        zero_order_beliefs = query_parameters['belief']['zero_order_belief'][-1]
        nested_beliefs = query_parameters['belief']['nested_beliefs'][-1]
        zero_order_beliefs = np.round(zero_order_beliefs, 3)
        nested_beliefs = np.round(nested_beliefs, 3)
        p1 = zero_order_beliefs[0]
        p2 = zero_order_beliefs[1]
        q1 = nested_beliefs[0]
        q2 = nested_beliefs[1]
        q3 = nested_beliefs[2]
        if self.config.get_from_general("number_of_rational_agents"):
            results = self.data.loc[(self.data['trial_number'] == trial) &
                                    (self.data['sender_threshold'] == threshold) &
                                    (self.data['p1'] == p1) & (self.data['p2'] == p2) &
                                    (self.data['q1'] == q1) & (self.data['q2'] == q2) & (self.data['q3'] == q3)]
        else:
            p3 = zero_order_beliefs[2]
            results = self.data.loc[
                (self.data['trial_number'] == trial) & (self.data['sender_threshold'] == threshold) &
                (self.data['p1'] == p1) & (self.data['p2'] == p2) & (self.data['p3'] == p3)]
        try:
            q_values = results.groupby('action')['q_value'].mean().reset_index()
        except DataError:
            print(f'Missing numeric output for query:trial={trial},threshold={threshold},p1={p1},p2={p2})')
            q_values = pd.DataFrame()
        return q_values

    def update_table(self, q_values: np.array, history: np.array, beliefs: np.array, game_parameters: dict):
        """
        This method add new Q-value to
        :param q_values:
        :param history:
        :param beliefs:
        :param game_parameters:
        :return:
        """
        trial_data = np.array([game_parameters['trial'], game_parameters['seed'], game_parameters['threshold']])
        zero_order_belief = np.round(beliefs["zero_order_belief"][-1, :], 3)
        nested_beliefs = np.round(beliefs["nested_beliefs"][-1, :], 3)
        data_to_append = pd.DataFrame(np.c_[q_values,
                                            np.tile(trial_data, (q_values.shape[0], 1)),
                                            np.tile(zero_order_belief, (q_values.shape[0], 1)),
                                            np.tile(nested_beliefs, (q_values.shape[0], 1)),
                                            np.tile(history, (q_values.shape[0], 1))],
                                      columns=self.columns)
        self.new_data = pd.concat([self.new_data, data_to_append])
        self.data = pd.concat([self.data, data_to_append])
        self.update_buffer_data(data_to_append)

    def _get_memoization_columns(self):
        if self.config.get_from_general("number_of_rational_agents") == 1:
            columns = ["action", "q_value", "trial_number", "seed", "sender_threshold", "p1", "p2",
                       "q1", "q2", "offer", "response"]
        else:
            columns = ["action", "q_value", "trial_number", "seed", "sender_threshold", "p1", "p2",
                       "q1", "q2", "q3", "offer", "response"]
        return columns


