import numpy as np
from scipy.misc import toimage

class State2Image:
    # return one information (1- image(s) representation for no reads)
    def getInitialState(self):
        pass

    # return two information (1- image(s) representation for the reads and 2- PM for the reads)
    def getStateInfoForReads(self, reads):
        pass

    def __init__(self, match, mismatch, gap, reads, max_read_len = None, number_of_reads = None):
        self.reads = reads
        self.number_of_reads = number_of_reads if number_of_reads is not None else len(reads)
        if max_read_len is None:
            for read in reads:
                 if max_read_len is None or len(read) > max_read_len:
                        max_read_len = len(read)
        self.max_read_len = max_read_len
        self.image_width = self._getMaxWidth()
        self.image_height = self.number_of_reads
        self.match = match
        self.mismatch = mismatch
        self.gap = gap
        self.clearBuffer()

    # clear two buffers which mantain overlap values already computed as well as
    # partial images that have already been built
    def clearBuffer(self):
        self.overlap_positions = {}
        self.root_image_search = {"row" : -1, "col" : 0, "read_id" : -1, "pm" : 0.0}
        self.sw_scores = {}

    # return the maximum width for this object
    def _getMaxWidth(self):
        return self.getMaxWidth(self.number_of_reads, self.max_read_len)

    # identify in which position two reads overlap exactly
    # if no overlap exists, -1 is returned
    # (to improve performance, use getOverlapPosition instead)
    def _resolveOverlapPosition(self, from_read_id, to_read_id):
        if from_read_id != -1:
            cur_pos = None
            read1 = self.reads[from_read_id]
            read2 = self.reads[to_read_id]
            for i in range(len(read1)):
                p1 = read1[i:]
                p2 = read2[:len(p1)]
                if p1 == p2:
                    return i
        return -1

    # identify in which position two reads overlap exactly
    # if no overlap exists, -1 is returned
    def getOverlapPosition(self, from_read_id, to_read_id):
        if from_read_id not in self.overlap_positions or to_read_id not in self.overlap_positions[from_read_id]:
            self._addOverlapPosition(from_read_id, to_read_id, self._resolveOverlapPosition(from_read_id, to_read_id))
        return self.overlap_positions[from_read_id][to_read_id]

    # store an overlap computed to further queries
    def _addOverlapPosition(self, from_read_id, to_read_id, position):
        if not from_read_id in self.overlap_positions:
            self.overlap_positions[from_read_id] = {}
        self.overlap_positions[from_read_id][to_read_id] = position

    # create a compressed image for given reads, identified in the list read_ids
    # by their positions at self.reads
    # (to improve performance, this function uses a preffix tree that stores
    # all images built, so that, if the image corresponding to the reads with
    # ids 5, 2 and 7 has been already built, the construction of the image
    # corresponding to reads 5, 2, 7 and 3 need only to produce the pixels
    # of the last row, because all others have already been produced)
    def _getCompressedImageForReads(self, read_ids):
        image = self.getEmptyCompressedImage(self.image_height)
        cur_state = self.root_image_search
        if type(read_ids) == list:
            aux_repeat = set()
            repeat = False
            for read_id in read_ids:
                if not repeat:
                    repeat = read_id in aux_repeat
                cur_state = self._getNextState(cur_state, read_id, repeat)
                aux_repeat.add(read_id)
                image[cur_state["row"]] = (cur_state["col"], cur_state["pixels"])
        return image, cur_state["pm"]

    # navigate within the preffix tree to find the next node considering
    # the current node and the next read to be incorporated (identified by
    # its order in self.reads)
    def _getNextState(self, cur_state, next_read_id, repeat):
        if next_read_id not in cur_state:
            next_state = {}
            cur_state[next_read_id] = next_state
            next_state["read_id"] = next_read_id
            next_state["row"] = cur_state["row"] + 1
            cur_read_id = cur_state["read_id"]
            overlap = self.getOverlapPosition(cur_read_id, next_read_id)
            cur_read_len = 1 if cur_read_id == -1 else len(self.reads[cur_read_id])
            #offset = overlap if overlap != -1 else 0
            #next_state["col"] = cur_state["col"] + offset
            next_state["col"] = 0 if overlap == -1 else cur_state["col"] + overlap
            next_state["pixels"] = [self._getPixelValue(nucleotide) for nucleotide in self.reads[next_read_id]]
            next_state["pixels"] = np.array(next_state["pixels"], dtype = np.uint8)
            next_state["pm"] = cur_state["pm"] + (self.sw(cur_read_id, next_read_id) if cur_read_id != next_read_id else 0)
            # next_state["pm"] = 0 if repeat else cur_state["pm"] + self.sw(cur_read_id, next_read_id)
            # next_state["pm"] *= 2 if next_state["row"] + 1 == self.number_of_reads and not repeat else 1
        return cur_state[next_read_id]

    # calculates each instance of a recurrent function to calculate sequences overlaps
    def getOverlapValue(self, i, j, matrix, s1, s2):
        score = self.match if s1[i-1] == s2[j-1] else self.mismatch
        aux = max(
            matrix[i-1][j-1] + score,
            matrix[i-1][j] + self.gap,
            matrix[i][j-1] + self.gap,
            0
        )
        return aux

    # define a pixel value for each nucleotide
    def _getPixelValue(self, nucleotide):
        return ['A', 'C', 'G', 'T'].index(nucleotide) * 30 + 135

    # calculate Smith-Waterman score for two reads (identified by their
    # positions at self.reads)
    # (to improve performance, use sw instead)
    def _sw(self, from_read_id, to_read_id):
        s1 = self.reads[from_read_id]
        s2 = self.reads[to_read_id]
        l = len(s1)+1
        c = len(s2)+1
        max_value = 0.0
        max_indexes = []
        matrix = np.array([0.0 for _ in range(l * c)]).reshape(l, c)
        for i in range(1, l):
            for j in range(1, c):
                matrix[i][j] = self.getOverlapValue(i, j, matrix, s1, s2)
                if matrix[i][j] > max_value:
                    max_value = matrix[i][j]
                    max_indexes = [(i, j)]
                elif max_value > 0.0 and max_value == matrix[i][j]:
                    max_indexes.append((i, j))

        for x in max_indexes:
            if x[0] >= x[1]:
                return max_value

        return 0

    # calculate Smith-Waterman score for two reads (identified by their
    # positions at self.reads)
    def sw(self, from_read_id, to_read_id):
        if from_read_id == -1:
            return 0.0
        l = from_read_id
        r = to_read_id

        unseen = False
        if l not in self.sw_scores:
            self.sw_scores[l] = {}
            unseen = True
        if r not in self.sw_scores[l]:
            unseen = True

        if unseen:
            self.sw_scores[l][r] = self._sw(l, r)

        return self.sw_scores[l][r]

    # define the maximum width the image representing one state can have
    # this maximum width happens when all reads overlap in only one nucleotide
    @staticmethod
    def getMaxWidth(number_of_reads, max_read_len):
        return max_read_len + (number_of_reads - 1) * (max_read_len - 1)

    # translate the compressed structure that represents an image to a regular image
    # (matrix format, where rows correspond to height and columns to width)
    # specification of compressed format:
    #    Each row of the image is represented as a 2-tuple, being the first
    #        element the column where the corresponding 'read' starts to be drawed
    #        and the second element corresponds to a list containing the pixel
    #        values to visually represent such 'read'
    @staticmethod
    def decompressImage(compressed, image_width, image_height):
        image = [[255 for _ in range(image_width)] for _ in range(image_height)]
        for i in range(image_height):
            start_col = compressed[i][0]
            if (start_col == -1):
                continue
            pixels = compressed[i][1]
            for j in range(len(pixels)):
                image[i][start_col + j] = pixels[j]
        return image

    # get one compressed image and save it as an image file
    @staticmethod
    def saveCompressedImage(image, image_width, image_height, output_file):
        State2Image.saveImage(State2Image.decompressImage(image, image_width, image_height), output_file)

    # save an image as an image file
    @staticmethod
    def saveImage(image, output_file):
        array = np.array(image)
        toimage(array, cmin=0, cmax=255).save(output_file)

    # return a compressed empty image
    @staticmethod
    def getEmptyCompressedImage(image_height):
        return [(-1, None) for _ in range(image_height)]
