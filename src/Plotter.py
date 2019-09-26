import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, maxPM, output_path):
        self.maxPM = maxPM
        self.output_path = output_path
        self.clearBuffer()
        
    def clearBuffer(self):
        self.pms = []
        self.epsilons = []
        
    def addPoint(self, pm, epsilon):
        self.epsilons.append(epsilon)
        self.pms.append(pm / self.maxPM)
        
    def plotPerformance(self):
        x = range(len(self.epsilons))
        plt.plot(x, self.epsilons)
        plt.plot(x, self.pms)
        plt.savefig(self.output_path)
