from State2ForwardImage import State2ForwardImage
import numpy as np

class State2HiddenForwardImage(State2ForwardImage):
    def __init__(self, reads, max_read_len, match, mismatch, gap, n_reads, nucleotides_in_grayscale, reward_system):
        super().__init__(reads, max_read_len, match, mismatch, gap, n_reads, nucleotides_in_grayscale, reward_system)

    # return two information (1- image(s) representation for the reads and 2- PM for the reads)
    def getStateInfoForReads(self, read_ids_order):
        image1, info = super().getStateInfoForReads(read_ids_order)
        image1 = image1[0]
        hide = False
        for i in range(1,len(image1)):
            if not hide and image1[i][0] == 0:
                aux = min(len(image1[i][1]), len(image1[i-1][1]))
                if not np.array_equal(image1[i][1][:aux], image1[i-1][1][:aux]):
                    hide = True
            if hide:
                image1[i] = (-1, None)
        return [image1], info
