from iug_task import *
from agent_factory import *
import argparse
import os

def export_beliefs_to_file(data: Union[pd.DataFrame, dict], directory_name, output_directory, nested=False):
    if not nested:
        outdir = os.path.join(config.beliefs_dir, directory_name)
    else:
        outdir = directory_name
    if type(data) == dict:
        for table_name in data.keys():
            new_directory_name = os.path.join(outdir, table_name)
            export_beliefs_to_file(data[table_name], new_directory_name, output_directory, True)
        return None
    if data is not None:
        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
        # Bypass Windows path limit for belief files too
        safe_path = "\\\\?\\" + os.path.abspath(os.path.join(outdir, output_directory))
        data.to_csv(safe_path, index=False)


def set_experiment_name(receivers_threshold, senders_threshold):
    return f'receiver_gamma_{receivers_threshold}_sender_gamma_{senders_threshold}'


def simulate_iug_task(sender_agent, receiver_agent, senders_threshold, receivers_threshold):
    sender_agent.threshold = senders_threshold
    receiver_agent.threshold = receivers_threshold
    # Initial experiment name
    experiment_name = set_experiment_name(receiver_agent.threshold, sender_agent.threshold)
    config.new_experiment_name(experiment_name)
    print(f'Sender parameters: gamma = {sender_agent.threshold}', flush=True)
    print(f'Receiver parameters: gamma = {receiver_agent.threshold}', flush=True)
    experiment_results, q_values, receiver_belief, sender_belief, receiver_mental_state = \
        eat_task_simulator.simulate_task(sender_agent, receiver_agent, receiver_agent.threshold, sender_agent.threshold)
    if sender_agent.name == "DoM(1)_sender" and config.get_from_general("use_memoization"):
        sender_agent.memoization_table.save_data()
    sender_agent.reset(terminal=True)
    receiver_agent.reset(terminal=True)
    experiment_name = config.experiment_name
    if config.args.save_results == "True":
        output_file_name = f'{experiment_name}_seed_{config.seed}.csv'
        
       # 1. Force the computer to create both folders before saving
        os.makedirs(config.simulation_results_dir, exist_ok=True)
        os.makedirs(config.q_values_results_dir, exist_ok=True)
        
        # 2. Safely save the files by bypassing the Windows path length limit
        sim_path = "\\\\?\\" + os.path.abspath(os.path.join(config.simulation_results_dir, output_file_name))
        q_path = "\\\\?\\" + os.path.abspath(os.path.join(config.q_values_results_dir, output_file_name))
        experiment_results.to_csv(sim_path, index=False)
        q_values.to_csv(q_path, index=False)
        
        export_beliefs_to_file(receiver_belief, 'receiver_beliefs', output_file_name)
        export_beliefs_to_file(sender_belief, 'sender_beliefs', output_file_name)
        export_beliefs_to_file(receiver_mental_state, 'receiver_mental_state', output_file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cognitive hierarchy task')
    parser.add_argument('--environment', type=str, default='first_task', metavar='N',
                        help='game environment (default: first_task)')
    parser.add_argument('--seed', type=int, default='6431', metavar='N',
                        help='set simulation seed (default: 6431)')
    parser.add_argument('--sender_tom', type=str, default='DoM0', metavar='N',
                        help='set rational_sender tom level (default: DoM0)')
    parser.add_argument('--receiver_tom', type=str, default='DoM0', metavar='N',
                        help='set rational_receiver tom level (default: DoM0)')
    parser.add_argument('--softmax_temp', type=float, default='0.05', metavar='N',
                        help='set softmax temp (default: 0.05)')
    parser.add_argument('--senders_threshold', type=float, default='0.0', metavar='N',
                        help='set senders threshold (default: 0.0)')
    parser.add_argument('--receivers_threshold', type=float, default='0.0', metavar='N',
                        help='set receivers threshold (default: 0.0)')
    parser.add_argument('--save_results', type=str, default='True', metavar='N',
                        help='save simulation results (default: True)')
    args = parser.parse_args()
    config = init_config(args.environment, args)
    factory = AgentFactory()
    rational_sender = factory.constructor("rational_sender")
    random_sender = factory.constructor("rational_sender", "DoM-1")
    rational_receiver = factory.constructor("rational_receiver")
    experiment_data = factory.create_experiment_grid()
    report_point = factory.grid_size
    sender_parameters = experiment_data["sender_parameters"]
    receiver_parameters = experiment_data["receiver_parameters"]
    i = 0
    np.random.seed(config.seed)
    eat_task_simulator = IUG(config.seed)
    if config.get_from_general("state") == 'debug':
        simulate_iug_task(rational_sender, rational_receiver, args.senders_threshold, args.receivers_threshold)
    else:
        for sender_threshold in sender_parameters:
            for receiver_threshold in receiver_parameters:
                # set random senders
                eat_task_simulator.reset()
                if sender_threshold == 0:
                    if config.get_from_general("skip_random"):
                        continue
                    sender = random_sender
                else:
                    if config.get_from_general("skip_rational"):
                        continue
                    sender = rational_sender
                receiver = rational_receiver
                # Update individual parameters
                simulate_iug_task(sender, receiver, sender_threshold, receiver_threshold)
                print("#" * 10 + ' simulation over ' + "#" * 10, flush=True)
                i += 1
                print(f'{i / factory.grid_size * 100}% of trials completed', flush=True)
