from State2Image import State2Image

import random

class State2RandomImage(State2Image):
    def __init__(self, reads, max_read_len, match, mismatch, gap, n_reads):
        super().__init__(match, mismatch, gap, reads, max_read_len, n_reads)
        self.init = None

    def countFramesPerState(self):
        return 1

    # return two empty images that correspond to the initial state of
    # state space
    def getInitialState(self):
        if self.init is None:
            aux = State2Image.getEmptyCompressedImage(self.image_height)
            self.init = [aux]
        return self.init

    # return two information (1- image(s) representation for the reads and 2- PM for the reads)
    def getStateInfoForReads(self, read_ids_order):
        o = read_ids_order[::-1] if (random.randint(0, 1) == 0) else read_ids_order
        image, info = self._getCompressedImageAndInfoForReads(o)
        return [image], info
