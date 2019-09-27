import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, maxPM, output_path, title = None, additionalLabels = None):
        self.maxPM = maxPM
        self.output_path = output_path
        self.title = title
        self.additionalLabels = additionalLabels
        self.clearBuffer()

    def clearBuffer(self):
        self.axis_x = []
        self.pms = []
        self.epsilons = []
        self.additional = []

    def addPoint(self, pm, epsilon, x = None, additional = None):
        if x is not None:
            self.axis_x.append(x)
        self.epsilons.append(epsilon)
        self.pms.append(pm / self.maxPM)
        if additional is not None:
            if type(additional)!=list:
                additional = [additional]
            self.additional.append(additional)

    def plotPerformance(self):
        if len(self.axis_x) != len(self.epsilons):
            x = range(len(self.epsilons))
        else:
            x = self.axis_x
        plt.plot(x, self.pms, label='Relative PM')
        plt.plot(x, self.epsilons, label='Epsilon')
        if len(self.additional) == len(self.epsilons):
            aux = len(self.additional[0])
            error = False
            for i in range(1, len(self.additional)):
                if len(self.additional[i]) != aux:
                    error = True
                    break
            if not error:
                for col in range(aux):
                    col_data = []
                    for i in range(len(self.additional)):
                        col_data.append(self.additional[i][col])
                    label_value = self.additionalLabels[col] if self.additionalLabels is not None and col < len(self.additionalLabels) else "Data " + str(col+1)
                    plt.plot(x, col_data, label=label_value)
        plt.legend()
        if self.title is not None:
            plt.suptitle(self.title)
        plt.savefig(self.output_path)
