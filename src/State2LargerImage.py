from State2Image import State2Image

class State2LargerImage(State2Image):
    def __init__(self, reads, max_read_len, match, mismatch, gap, n_reads, nucleotides_in_grayscale, reward_system):
        super().__init__(match, mismatch, gap, reads, max_read_len, n_reads, nucleotides_in_grayscale, reward_system)
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
        image1, aux = self._getCompressedImageAndInfoForReads(read_ids_order)
        aux1 = aux
        image2, aux = self._getCompressedImageAndInfoForReads(read_ids_order[::-1])
        aux2 = aux
        max_width = None
        max_image = None
        w1 = 0
        w2 = 0
        for row in range(self.image_height):

            s1 = image1[row][0]
            p1 = image1[row][1]
            s2 = image2[row][0]
            p2 = image2[row][1]
            w1 = 0 if (s1 == -1) else s1 + len(p1)
            w2 = 0 if (s2 == -1) else s2 + len(p2)

            if w1 > w2:
                if max_width is None or w1 > max_width:
                    max_width = w1
                    max_image = 1
            else:
                if max_width is None or w2 > max_width:
                    max_width = w2
                    max_image = 2
        if max_image == 1:
            image = image1
            info = aux1
        else:
            image = image2
            info = aux2
        return [image], info
