from Plotter import Plotter
import csv
import sys

if len(sys.argv) < 5:
    print("Usage: python3 " + sys.argv[0] + " <input.csv> <max_pm> <title> <output.png> [additional_labels]*")
    print("Example: python3 " + sys.argv[0] + " A_1_1.csv 55 Graphic saida_1_1.png 'stddev' 'avg'")
    sys.exit(1)

input_path = sys.argv[1]
additionalLabels = None if len(sys.argv) == 5 else sys.argv[5:]
plotter = Plotter(float(sys.argv[2]), sys.argv[4], sys.argv[3], additionalLabels)

with open(input_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        additional = None if len(row) == 3 else [float(val) for val in row[3:]] 
        plotter.addPoint(float(row[1]), float(row[2]), int(row[0]), additional)
plotter.plotPerformance()

