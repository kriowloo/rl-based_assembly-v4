import matplotlib.pyplot as plt

def plotPerformance(epsilons, pms, output_path):
    x = range(len(epsilons))
    plt.plot(x, epsilons)
    plt.plot(x, pms)
    plt.savefig(output_path)
