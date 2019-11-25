from State2Image import State2Image

class State2ForwardImages(State2Image):
    def __init__(self, reads, max_read_len, match, mismatch, gap, n_reads, number_of_frames, nucleotides_in_grayscale):
        super().__init__(match, mismatch, gap, reads, max_read_len, n_reads, nucleotides_in_grayscale)
        self.init = None
        self.number_of_frames = number_of_frames

    def countFramesPerState(self):
        return self.number_of_frames

    # return two empty images that correspond to the initial state of
    # state space
    def getInitialState(self):
        if self.init is None:
            aux = State2Image.getEmptyCompressedImage(self.image_height)
            self.init = [aux for _ in range(self.number_of_frames)]
        return self.init

    # return two information (1- image(s) representation for the reads and 2- PM for the reads)
    def getStateInfoForReads(self, read_ids_order):
        image1, info = self._getCompressedImageAndInfoForReads(read_ids_order)
        images = [image1]
        last_row = -1
        for i in range(self.image_height):
            if image1[i][0] != -1:
                last_row = i
        for _ in range(1, self.number_of_frames):
            cur_image = images[-1][:]
            if last_row >= 0:
                cur_image[last_row] = (-1, None)
                last_row -= 1
            images.append(cur_image)
        return images, info
