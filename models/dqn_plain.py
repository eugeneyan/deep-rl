import numpy as np
from keras.layers import Dense
from keras.losses import mse
from keras.models import Sequential
from keras.optimizers import Adam

from utils.losses import huber_loss


class DQNPlain:

    def __init__(self, inputs, outputs, hidden_layers, gamma, learning_rate, epsilon, epsilon_min, loss='mse'):
        self.input_size = inputs
        self.output_size = outputs
        self.gamma = gamma
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.regularization_factor = 0.01
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.loss = ''
        if loss == 'huber':
            self.loss = huber_loss
        elif loss == 'mse':
            self.loss = mse

        self.model = self.build_model()

    def build_model(self, ):
        model = Sequential()

        model.add(
            Dense(self.hidden_layers[0], input_dim=self.input_size, activation='relu'))

        for i in range(1, len(self.hidden_layers) - 1):
            layer_size = self.hidden_layers[i]
            model.add(Dense(layer_size, activation='relu'))

        model.add(Dense(self.output_size, activation='linear'))
        model.compile(loss=self.loss, optimizer=Adam(lr=self.learning_rate))
        return model

    def take_action(self, state, episode):
        adjusted_epsilon = max(self.epsilon_min, self.epsilon ** episode)
        values = self.model.predict(state.reshape(1, self.input_size))[0]

        if np.random.random() < adjusted_epsilon:
            action = np.random.randint(len(values))
        else:
            action = np.argmax(values)

        return action, adjusted_epsilon

    def train_minibatch(self, minibatch):
        # extract values by type from the minibatch
        state_batch = np.array(minibatch[:, 0].tolist())
        action_batch = minibatch[:, 1].astype(int)
        rewards_batch = np.array(minibatch[:, 2])
        state_prime_batch = np.array(minibatch[:, 3].tolist())
        is_terminal_batch = minibatch[:, 4].astype(bool)

        state_value_batch = self.model.predict(state_batch)
        next_state_value_batch = self.model.predict(state_prime_batch)

        state_value_batch[range(len(state_value_batch)), action_batch] = rewards_batch + self.gamma * np.amax(
            next_state_value_batch, axis=1) * ~is_terminal_batch

        # update the neural network weights
        self.model.train_on_batch(state_batch, state_value_batch)

    def save_model(self, path):
        self.model.save_weights(path)

    def load_model(self, path):
        self.model.load_weights(path)
