import glob
from PIL import Image
import numpy as np
from keras.models import Sequential, clone_model
from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras.optimizers import RMSprop
import sys
from keras import backend as K
import tensorflow as tf
from keras.backend import tensorflow_backend
import random

def getArray(images):
    array = []
    for image in images:
        im = Image.open(image)
        array.append(np.array(im))
    return array

def getModel(image_height, image_width, frames_per_state = 1):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=3, activation='relu', input_shape=(image_height, image_width, frames_per_state)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(image_height))
    optimizer = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    model.compile(optimizer, loss='mse')
    return model

def train(left_frames, action, right_frames, reward, n_epochs_per_register, test_inputs, gamma, models, frames_per_state, reset_after):
    left_images = getArray(left_frames)
    right_images = getArray(right_frames)
    h = left_images[0].shape[0]
    w = left_images[0].shape[1]
    if models is None:
        model = getModel(h, w, frames_per_state)
        models = [model, clone_model(model), 0]
    inspect(models[0], test_inputs, action)

    for i in range(1,n_epochs_per_register+1):
        backpropagation(models, left_images, action, right_images, reward, gamma, h, w, reset_after)
        inspect(models[0], test_inputs)
    return models

def getProbability(output, i):
    if np.max(output) - np.min(output) == 0:
        return 0.0
    return (output[i] - np.min(output)) / (np.max(output) - np.min(output))

def inspect(model, cnn_inputs, action = None):
    text = ""
    for i in range(len(cnn_inputs)):
        output = model.predict(cnn_inputs[i])[0]
        text += str(action) + "," + str(i) + "," + ("%.2f" % (getProbability(output, i))) + "," + ("%.4f" % (np.min(output))) + "," + ("%.4f" % (np.max(output))) + "\n"
    print(text, end = '')

def getTestCNNInputs(files, frames_per_state):
    inputsCNN = []
    for i in range(1, len(files)):
        images = []
        for k in range(frames_per_state):
            images.append(files[-1 if i+k >= len(files) else i + k])
        images = getArray(images)
        h = images[0].shape[0]
        w = images[0].shape[1]
        inputsCNN.append(imageToCNN(images, h, w))
    return inputsCNN

def backpropagation(models, left_images, action, right_images, reward, gamma, h, w, reset_after):
    target = reward
    if action != 0:
        target = reward + gamma * np.amax(models[1].predict(imageToCNN(right_images, h, w))[0])
    state_inputcnn = imageToCNN(left_images, h, w)
    target_f = models[0].predict(state_inputcnn)
    target_f[0][action] = target
    models[0].fit(state_inputcnn, target_f, epochs=1, verbose=0)
    models[2] += 1
    if models[2] == reset_after:
        models[1].set_weights(models[0].get_weights())
        models[2] = 0

def normalizePixels(image):
    return image / 255

def imageToCNN(images,h, w):
    bff = np.empty((1, h, w, len(images)))
    for k in range(len(images)):
        image = images[k]
        image = normalizePixels(image)
        for i in range(h):
            for j in range(w):
                bff[0][i][j][k] = image[i][j]
    return bff

def processFile(image):
    aux = image.split("%")
    action = int(aux[0])
    reward = float(aux[1].split(".")[0].replace("_","."))
    return action, reward

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 ", sys.argv[0], " <batches> <n_epochs_per_batch> <frames_per_state> <reset_after>")
        print()
        print("Example: python3 ", sys.argv[0]," 15 10 1 50  # run 15 batches (with 10 instances per class in each) using 1 frame per state and reseting after 50 updatings")
        sys.exit(1)

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    tensorflow_backend.set_session(session)

    files = glob.glob("*.png")
    files.sort()
    batches = int(sys.argv[1])
    n_epochs_per_batch = int(sys.argv[2])
    frames_per_state = int(sys.argv[3])
    reset_after = int(sys.argv[4])
    test_inputs = getTestCNNInputs(files, frames_per_state)
    models = None
    for _ in range(batches):
        indexes = list(range(len(files) - 1))
        random.shuffle(indexes)
        for i in indexes:
            right_images = []
            left_images = []
            for k in range(frames_per_state):
                if i + k >= len(files):
                    right_images.append(files[-1])
                else:
                    right_images.append(files[i+k])
                if i + k + 1 >= len(files):
                    left_images.append(files[-1])
                else:
                    left_images.append(files[i+k+1])
            action, reward = processFile(right_images[0])
            models = train(left_images, action, right_images,reward, n_epochs_per_batch, test_inputs, 0.995, models, frames_per_state, reset_after)
    for i in range(len(test_inputs)):
        output = models[0].predict(test_inputs[-1-i])[0]
        print(np.argmax(output), output)
