import csv
import sys
import math

def processBatch(output_data, episode, epsilon, batch):
    if len(batch) == 0:
        return
    aux = []
    for b in batch:
        for x in b:
            aux.append(x)
    batch = aux
    average = 0.0
    for v in batch:
        average += v
    average /= len(batch)
    std = 0.0
    for v in batch:
        std += (v - average) ** 2
    std /= len(batch)
    std = math.sqrt(std)
    std /= average
    output_data.append([episode, average, epsilon, std])


if len(sys.argv) < 4:
    print("Usage: python3 " + sys.argv[0] + " <batch_size> <output.csv> <input_file1> [<input_file2>]*")
    print("Example: python3 " + sys.argv[0] + " 5 outputA.csv inputA1.csv inputA2.csv inputA3.csv inputA4.csv")
    sys.exit(1)

batch_size = int(sys.argv[1])
output_path = sys.argv[2]
input_files = sys.argv[3:]
data = []
for input_path in input_files:
    file_data = []
    data.append(file_data)
    with open(input_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            file_data.append(row)

rows = len(data[0])
files_count = len(data)
for i in range(1, files_count):
    if len(data[i]) != rows:
        print("Number of rows in all files doesn't match.")
        sys.exit(1)

output_data = []
batch = []
for i in range(rows):
    row = []
    for j in range(files_count):
        row.append(float(data[j][i][1]))
    batch.append(row)
    if len(batch) == batch_size:
        processBatch(output_data, i+1, float(data[0][i][2]), batch)
        batch = []
processBatch(output_data, rows, float(data[0][rows-1][2]), batch)
with open(output_path, mode='w') as output_file:
    writer = csv.writer(output_file, delimiter=',')
    for row in output_data:
        writer.writerow(row)
