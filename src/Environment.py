import os
from State2Image import State2Image

class Environment:
    def __init__(self, ol, reads, number_of_reads = None):
        self.ol = ol
        self.reads = reads
        self.number_of_reads = len(self.reads) if number_of_reads is None else number_of_reads
        self.actions_taken = []

    def getInitialState(self):
        self.actions_taken = []
        return self.ol.getInitialState()

    def step(self, action):
        self.actions_taken.append(action)
        img, pm = self.ol.getStateInfoForReads(self.actions_taken)
        stop = len(self.actions_taken) == self.number_of_reads
        # final = len(set(self.actions_taken)) == self.number_of_reads
        # reward = 0.1 if not final else pm
        next_state = img
        reward = pm
        # self.debug(next_state)
        return next_state, reward, stop

    def debug(self, next_state):
        token = '/data/debug.txt'
        if os.path.exists(token) and len(self.actions_taken) > 8:
            os.remove(token)
            prefix = "/data/" + "-".join(str(x) for x in self.actions_taken) + "_"
            for i in range(len(next_state)):
                output_file = prefix + str(i) + ".png"
                image = next_state[i]
                State2Image.saveCompressedImage(image, self.ol.image_width, self.ol.image_height, output_file)
