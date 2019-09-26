from State2Image import State2Image

class State2HiddenForwardImage(State2Image):
    def __init__(self, reads, max_read_len, match, mismatch, gap, n_reads):
        super().__init__(match, mismatch, gap, reads, max_read_len, n_reads)
        self.init = None

    # return two empty images that correspond to the initial state of
    # state space
    def getInitialState(self):
        if self.init is None:
            aux = State2Image.getEmptyCompressedImage(self.image_height)
            self.init = [aux]
        return self.init

    # return two information (1- image(s) representation for the reads and 2- PM for the reads)
    def getStateInfoForReads(self, read_ids_order):
        image1, pm1 = self._getCompressedImageForReads(read_ids_order)
        hide = False
        for i in range(len(image1)):
            if hide:
                image1[i][0] = -1
                image1[i][1] = None
            elif i > 0 and image1[i][0] == 0:
                aux = min(len(image1[i][1]), image1[i-1][1])
                if image1[i][1][:aux] != image1[i-1][1][:aux]:
                    hide = True
        return [image1], pm1
