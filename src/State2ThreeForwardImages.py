from State2ForwardImages import State2ForwardImages

class State2ThreeForwardImages(State2ForwardImages):
    def __init__(self, reads, max_read_len, match, mismatch, gap, n_reads):
        super().__init__(reads, max_read_len, match, mismatch, gap, n_reads, 3)

