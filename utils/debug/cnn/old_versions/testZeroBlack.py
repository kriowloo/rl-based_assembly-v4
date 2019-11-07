import glob
from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import RMSprop
import sys

def getArray(image):
    im = Image.open(image)
    return np.array(im)

def getModel(image_height, image_width):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=3, activation='relu', input_shape=(image_height, image_width, 1)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(image_height))
    optimizer = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    model.compile(optimizer, loss='mse')
    return model

def train(left_image, action, right_image, reward, n_epochs_per_register, test_inputs, gamma = 0.995, model = None):
    left_image = getArray(left_image)
    right_image = getArray(right_image)
    h = left_image.shape[0]
    w = left_image.shape[1]

    if model is None:
        model = getModel(h, w)
    inspect(model, test_inputs, action)
    
    for i in range(1,n_epochs_per_register+1):
        backpropagation(model, left_image, action, right_image, reward, gamma, h, w)
        inspect(model, test_inputs)
    return model

def getProbability(output, i):
    if np.max(output) - np.min(output) == 0:
        return 0.0
    return (output[i] - np.min(output)) / (np.max(output) - np.min(output))

def inspect(model, cnn_inputs, action = None):
    for i in range(len(cnn_inputs)):
        output = model.predict(cnn_inputs[i])[0]
        print(str(action) + "," + str(i) + "," + str(getProbability(output, i)))

def getTestCNNInputs(files):
    inputsCNN = []
    for i in range(1, len(files)):
        image = getArray(files[i])
        h = image.shape[0]
        w = image.shape[1]
        inputsCNN.append(imageToCNN(image, h, w))
    return inputsCNN

def backpropagation(model, left_image, action, right_image, reward, gamma, h, w):
    target = reward
    if action != 0:
        target = reward + gamma * np.amax(model.predict(imageToCNN(right_image, h, w))[0])
    state_inputcnn = imageToCNN(left_image, h, w)
    target_f = model.predict(state_inputcnn)
    target_f[0][action] = target
    model.fit(state_inputcnn, target_f, epochs=1, verbose=0)

def imageToCNN(image,h, w):
    image = image / 255
    return np.array([image.reshape((h, w, 1))])

def processFile(image):
    aux = image.split("%")
    action = int(aux[0])
    reward = float(aux[1].split(".")[0].replace("_","."))
    return action, reward

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 ", sys.argv[0], " <batches> <n_epochs_per_batch>")
        print()
        print("Example: python3 ", sys.argv[0]," 10 10")
        sys.exit(1)
    files = glob.glob("*.png")
    files.sort()
    batches = int(sys.argv[1])
    n_epochs_per_batch = int(sys.argv[2])
    test_inputs = getTestCNNInputs(files)
    model = None
    for _ in range(batches):
        for i in range(len(files) - 1):
            right_image = files[i]
            left_image = files[i+1]
            action, reward = processFile(right_image)
            model = train(left_image, action, right_image,reward, n_epochs_per_batch, test_inputs, model = model)
        

