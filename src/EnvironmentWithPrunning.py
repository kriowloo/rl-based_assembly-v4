from Environment import Environment

class EnvironmentWithPrunning(Environment):
    def __init__(self, ol, reads, number_of_reads = None):
        super().__init__(ol, reads, number_of_reads)

    def isStopable(self, img, info):
        return super().isStopable(img, info) or info["breaks"] > 0
