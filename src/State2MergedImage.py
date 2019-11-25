from State2Image import State2Image

class State2MergedImage(State2Image):
    def __init__(self, reads, max_read_len, match, mismatch, gap, n_reads, nucleotides_in_grayscale):
        super().__init__(match, mismatch, gap, reads, max_read_len, n_reads, nucleotides_in_grayscale)
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
        zip1, info = self._getCompressedImageAndInfoForReads(read_ids_order)
        zip2, _ = self._getCompressedImageAndInfoForReads(read_ids_order[::-1])
        img1 = State2Image.decompressImage(zip1, self.image_width, self.image_height)
        img2 = State2Image.decompressImage(zip2, self.image_width, self.image_height)

        graytones = ['#A', '#C', '#G', '#T','A#', 'AA', 'AC', 'AG', 'AT','C#', 'CA', 'CC', 'CG', 'CT','G#', 'GA', 'GC', 'GG', 'GT','T#', 'TA', 'TC', 'TG', 'TT']
        conversion = {
            self._getPixelValue('A') : 'A',
            self._getPixelValue('C') : 'C',
            self._getPixelValue('G') : 'G',
            self._getPixelValue('T') : 'T',
        }

        image = []
        for row in range(self.image_height):
            s1 = zip1[row][0]
            s2 = zip2[row][0]
            p1 = zip1[row][1]
            p2 = zip2[row][1]
            l1 = 0 if p1 is None else len(p1)
            l2 = 0 if p2 is None else len(p2)
            s = s1 if s1 < s2 else s2
            pixels = []
            i1 = 0
            i2 = 0
            if s1 < s2:
                while True:
                    pixel1, pixel2 = None, None
                    if i1 < l1:
                        pixel1 = p1[i1]
                        i1 += 1

                    if s1 >= s2 and i2 < l2:
                        pixel2 = p2[i2]
                        i2 += 1

                    pixel1 = '#' if pixel1 is None else conversion[pixel1]
                    pixel2 = '#' if pixel2 is None else conversion[pixel2]

                    pixel = pixel1 + pixel2
                    pixel = 255 if pixel == '##' else graytones.index(pixel) * 10
                    pixels.append(pixel)

                    s1 += 1
                    if i1 >= l1 and i2 >= l2:
                        break
            else:
                while True:
                    pixel1, pixel2 = None, None
                    if i2 < l2:
                        pixel2 = p2[i2]
                        i2 += 1

                    if s2 >= s1 and i1 < l1:
                        pixel1 = p1[i1]
                        i1 += 1

                    pixel1 = '#' if pixel1 is None else conversion[pixel1]
                    pixel2 = '#' if pixel2 is None else conversion[pixel2]

                    pixel = pixel1 + pixel2
                    pixel = 255 if pixel == '##' else graytones.index(pixel) * 10
                    pixels.append(pixel)

                    s2 += 1
                    if i1 >= l1 and i2 >= l2:
                        break
            image.append((s, pixels))

        # merge
        info["breaks"] = 0
        return [image], info

