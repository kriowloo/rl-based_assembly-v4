from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Lambda
from keras.optimizers import RMSprop
from numpy.random import randint, seed, rand
import numpy as np
from collections import deque
import random
from keras import backend as K
import tensorflow as tf
from keras.backend import tensorflow_backend
from State2Image import State2Image
from Plotter import Plotter
import os

#TODO: 
# 1) Retirar Plotter de dentro deste código
# 2) Passar Plotter para utils
# 3) Implementar aqui apenas a possibilidade de exportar desempenho para CSV (existe um exemplo em src/tmp hardcoded para usar de exemplo)
# 4) Documentar programas de utils/tmp (eles servem para gerar plots e para resumir os graficos)

class DFADeepQNetwork:
    def _setupKerasCPU(self):
        K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=self.threads, inter_op_parallelism_threads=self.threads)))

    def _setupKerasGPU(self):
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        session = tf.Session(config=config)
        tensorflow_backend.set_session(session)

    def __init__(self, n_reads, max_read_len, frames_per_state, buffer_maxlen, epsilon, epsilon_decay, epsilon_min, gamma, threads, env, gpu_enabled = False, pixel_norm_type = 1, plot_fig_path = None, maxPM = None):
        self.env = env
        self.n_reads = n_reads
        self.max_read_len = max_read_len
        self.gpu_enabled = gpu_enabled
        self.threads = threads
        self.normalizePixel, self.getWhitePixelValue = self.getPixelNormalizationFunctions(pixel_norm_type)
        # (plot_fig_path = path to plot the performance figure; is None, no figure is saved)
        self.plotter = None if plot_fig_path is None or maxPM is None or maxPM <= 0 else Plotter(maxPM, plot_fig_path)

        # image_height: height of images that will represent states (ie: number of reads)
        self.image_height = n_reads
        # image_width: maximum width of images that will represent states
        self.image_width = State2Image.getMaxWidth(self.n_reads, self.max_read_len)
        # memory for replay
        self.memory = deque(maxlen=buffer_maxlen)

        # number of image frames for each state
        self.frames_per_state = frames_per_state

        # e-greedy parameters
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # q-learning parameter
        self.gamma = gamma

        # build CNN
        self.model = self._build_model()

    def normalizeWhiteToBlack(self, pixel_value):
        return (255 - pixel_value) / 255.0

    def normalizeBlackToWhite(self, pixel_value):
        return pixel_value / 255.0
        
    def getPixelNormalizationFunctions(self, pixel_norm_type):
        if pixel_norm_type == 1:
            return self.normalizeWhiteToBlack, lambda : 0
        return self.normalizeBlackToWhite, lambda : 1
        
    def _build_model(self):
        # setting up Keras to perform on CPU or GPU
        if self.gpu_enabled:
            self._setupKerasGPU()
        else:
            self._setupKerasCPU()

        # create model
        model = Sequential()
        # considering data_format=channel_last

        model.add(Conv2D(16, kernel_size=3, activation='relu', input_shape=(self.n_reads, self.image_width, self.frames_per_state)))
        model.add(Conv2D(32, kernel_size=3, activation='relu'))
        # model.add(Conv2D(16, kernel_size=8, strides=(4,4), activation='relu', input_shape=(self.n_reads, self.image_width, self.frames_per_state)))
        # model.add(Conv2D(32, kernel_size=4, strides=(2,2), activation='relu'))
        # model.add(Conv2D(4, kernel_size=8, activation='relu', input_shape=(self.image_width, self.n_reads,self.frames_per_state)))
        # model.add(Conv2D(8, kernel_size=4, activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        # model.add(Dense(128, activation='relu'))
        # model.add(Dense(self.n_reads, activation='softmax'))
        model.add(Dense(self.n_reads))
        optimizer = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        model.compile(optimizer, loss='mse')
        return model

    # define which action is going to be taken at the next step (e-greedy)
    def _act(self, state, training = True):
        if training and np.random.rand() <= self.epsilon:
            return self.env.getActionFromExploration()
        act_values = self.model.predict(self._stateToCNN(state))
        # return np.argmax(act_values[0])
        nn_outputs = act_values[0]
        max_action_id = None
        max_action_value = None
        for i in range(self.n_reads):
            if not i in self.env.actions_taken:
                val = nn_outputs[i]
                if max_action_id is None or val > max_action_value:
                    max_action_id = i
                    max_action_value = val
        return max_action_id

    # store each action taken in a limited memory
    def _remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # select randomly some actions taken and proceed to update the neural network
    def _replay(self, batch_size, run_with_unfilled_batch = False):
        if (len(self.memory) < batch_size):
            if not run_with_unfilled_batch:
                return
            batch_size = len(self.memory)
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(self._stateToCNN(next_state))[0])
            state_inputcnn = self._stateToCNN(state)
            target_f = self.model.predict(state_inputcnn)
            target_f[0][action] = target
            self.model.fit(state_inputcnn, target_f, epochs=1, verbose=0)

    # transform compressed image to a numpy array compatible with keras (data_format=channel_last)
    def _stateToCNN(self, state):
        images = []
        for h in range(self.image_height):
            aux_h = []
            images.append(aux_h)
            for w in range(self.image_width):
                aux_w = [self.getWhitePixelValue() for _ in range(len(state))]
                aux_h.append(aux_w)
        images = [images]
        for f in range(len(state)):
            compressed = state[f]
            for i in range(self.image_height):
                start_col = compressed[i][0]
                if (start_col == -1):
                    continue
                pixels = compressed[i][1]
                for j in range(len(pixels)):
                    images[0][i][start_col + j][f] = self.normalizePixel(pixels[j])
        array = np.array(images)
        # self.debug(state, array)
        return array

    def debug(self, state, array):
        token = '/data/debug.txt'
        if os.path.exists(token):
            os.remove(token)
            error = 0
            for i in range(len(state)):
                image = state[i]
                im = State2Image.decompressImage(image, self.image_width, self.image_height)
                for h in range(self.image_height):
                    for w in range(self.image_width):
                        if (255 - im[h][w])/255.0 != array[0][h][w][i]:
                            error += 1
                            print(str(array[0][h][w][i]) + " " + str((255 - im[h][w])/255.0))
            if error > 0:
                print("Erro " + str(error) + " w=" + str(self.image_width) + " h=" + str(self.image_height))
            else:
                print("OK")

    # decay epsilon to diminish the probability of taking random actions
    def _decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # train agent 
    def train(self, episodes, buffer_batch_size, max_actions_per_episode):
        # Iterate learning
        for e in range(episodes):
            # reset state in the beginning of each episode
            state = self.env.getInitialState()
            print("episode: " + str(e+1))
            time_t = 1 # action counter of the episode
            while True:
                action = self._act(state)


                # Advance the environment to the next state based on the action.
                next_state, reward, stop = self.env.step(action)

                print(str(action) + " " + str(reward))

                self._remember(state, action, reward, next_state, stop)
                state = next_state

                if stop or (max_actions_per_episode > 0 and time_t >= max_actions_per_episode):
                    # print episode id and the number of actions taken on it
                    print("episode: {}/{}, reward: {}, epsilon: {}".format(e+1, episodes, reward, self.epsilon))
                    if self.plotter is not None:
                        self.plotter.addPoint(reward, self.epsilon)
                    break
                time_t += 1
            # train the agent with the experience of the episode
            self._replay(buffer_batch_size)
            self._decay_epsilon()
        if self.plotter is not None:
            self.plotter.plotPerformance()
        return None

    # ask agent to take actions according its prior learning
    def test(self, n_actions):
        # reset state in the beginning of each episode
        state = self.env.getInitialState()
        action_counter = 0 # action counter of the episode
        stop = False
        while True:
            if action_counter >= n_actions or stop:
                break
            action = self._act(state, False)
            # Advance the environment to the next state based on the action.
            next_state, reward, stop = self.env.step(action)
            state = next_state
            action_counter += 1
            print(str(action) + " " + str(reward))
