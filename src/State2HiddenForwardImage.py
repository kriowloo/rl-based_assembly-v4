from State2ForwardImage import State2ForwardImage

class State2HiddenForwardImage(State2ForwardImage):
    def __init__(self, reads, max_read_len, match, mismatch, gap, n_reads):
        super().__init__(reads, max_read_len, match, mismatch, gap, n_reads)

    # return two information (1- image(s) representation for the reads and 2- PM for the reads)
    def getStateInfoForReads(self, read_ids_order):
        image1, pm1 = super().getStateInfoForReads(read_ids_order)
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
