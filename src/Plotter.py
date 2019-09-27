import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, maxPM, output_path, title = None):
        self.maxPM = maxPM
        self.output_path = output_path
        self.title = title
        self.clearBuffer()

    def clearBuffer(self):
        self.axis_x = []
        self.pms = []
        self.epsilons = []

    def addPoint(self, pm, epsilon, x = None):
        if x is not None:
            self.axis_x.append(x)
        self.epsilons.append(epsilon)
        self.pms.append(pm / self.maxPM)

    def plotPerformance(self):
        if len(self.axis_x) != len(self.epsilons):
            x = range(len(self.epsilons))
        else:
            x = self.axis_x
        plt.plot(x, self.epsilons, label='Epsilon')
        plt.plot(x, self.pms, label='Relative PM')
        plt.legend()
        if self.title is not None:
            plt.suptitle(self.title)
        plt.savefig(self.output_path)
