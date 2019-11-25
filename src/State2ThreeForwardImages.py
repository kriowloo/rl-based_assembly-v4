from State2ForwardImages import State2ForwardImages

class State2ThreeForwardImages(State2ForwardImages):
    def __init__(self, reads, max_read_len, match, mismatch, gap, n_reads, nucleotides_in_grayscale):
        super().__init__(reads, max_read_len, match, mismatch, gap, n_reads, 3, nucleotides_in_grayscale)

