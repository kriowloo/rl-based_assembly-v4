import os
from State2Image import State2Image
import random
import math

class Environment:
    def __init__(self, ol, reads, number_of_reads = None):
        self.ol = ol
        self.reads = reads
        self.number_of_reads = len(self.reads) if number_of_reads is None else number_of_reads
        self.actions_taken = []
        self.debug_episode = 0

    def debugImageGeneration(self, state, pm):
        pm = ("%.2f" % (pm)).replace('.','')
        while len(pm) <= 6:
            pm = "0" + pm
        e = str(self.debug_episode)
        while len(e) <= 10:
            e = "0" + e
        a = str(len(self.actions_taken))
        while len(a) <= 3:
            a = "0" + a
        for i in range(len(state)):
            output_file = "img_" + e + "_" + a + "_" + pm + "_" + str(i) + ".png" 
            State2Image.saveCompressedImage(state, self.ol.image_width, self.ol.image_height, output_file)
        
    def getInitialState(self):
        self.debug_episode += 1
        self.actions_taken = []
        state = self.ol.getInitialState()
        self.debugImageGeneration(state, 0)
        return state

    def step(self, action):
        self.actions_taken.append(action)
        img, pm = self.ol.getStateInfoForReads(self.actions_taken)
        self.debugImageGeneration(img, pm)
        stop = len(self.actions_taken) == self.number_of_reads
        # final = len(set(self.actions_taken)) == self.number_of_reads
        # reward = 0.1 if not final else pm
        next_state = img
        reward = pm
        # self.debug(next_state)
        return next_state, reward, stop

    def getActionFromExploration(self):
        # return self._getRandomActionWithoutRepeat()
        return self._getMaxRandomActionWithoutRepeat()

    def _getCandidateActions(self):
        candidates = []
        for action_id in range(self.number_of_reads):
            if not action_id in self.actions_taken:
                candidates.append(action_id)
        return candidates

    # select randomly an action that has not been selected before
    def _getRandomActionWithoutRepeat(self):
        candidates = self._getCandidateActions()
        return random.sample(candidates, 1)[0]

    def _getMaxRandomActionWithoutRepeat(self, percent = 1.0):
        if len(self.actions_taken) == 0:
            return random.randrange(0, self.number_of_reads)

        candidates = self._getCandidateActions()
        last = self.actions_taken[-1]
        n_candidates = math.ceil(self.number_of_reads * percent)
        if n_candidates > len(candidates):
            n_candidates = len(candidates)
        candidates = random.sample(candidates, n_candidates)
        max_action = None
        max_value = None
        for candidate in candidates:
            value = self.ol.sw(last, candidate)
            if max_action is None or value > max_value:
                max_value = value
                max_action = candidate
        return max_action

    def debug(self, next_state):
        token = '/data/debug.txt'
        if os.path.exists(token) and len(self.actions_taken) > 8:
            os.remove(token)
            prefix = "/data/" + "-".join(str(x) for x in self.actions_taken) + "_"
            for i in range(len(next_state)):
                output_file = prefix + str(i) + ".png"
                image = next_state[i]
                State2Image.saveCompressedImage(image, self.ol.image_width, self.ol.image_height, output_file)
