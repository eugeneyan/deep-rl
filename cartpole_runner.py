from argparse import ArgumentParser

from runner import runner

# Params for cart pole
n_episodes = 3000
epsilon = 0.99
epsilon_min = 0.003
batch_size = 128
gamma = 0.99
memory_bank_size = 5000
learning_rate = 0.005
update_rate = 10000
loss = 'mse'
ma_threshold = 195

# Initialize environment
env_name = 'CartPole-v0'

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', default='ddqn',
                        help='DeepRL model to use (options: dqn_plain, dqn, ddqn; default: %(default)s)')
    parser.add_argument('--render-env', default='y', help='Whether to render the environment (default: %(default)s)')
    parser.add_argument('--render-freq', type=int, default=100,
                        help='How frequently to render the env (default: %(default)s) '
                             '--render-env must be set to "y" to render environment')

    args = parser.parse_args()
    runner(env_name, memory_bank_size, batch_size, gamma, learning_rate,
           epsilon, epsilon_min, loss, n_episodes, ma_threshold, args)
