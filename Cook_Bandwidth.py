import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

BITS_IN_BYTE = 8.0
MBITS_IN_BITS = 1000000.0
MILLISECONDS_IN_SECONDS = 1000.0
LINK_FILE = "D:\BandwidthDataset\logs_all\\report_train_0001.log"

DATA_PATH = "D:\BandwidthDataset\logs_all\\"
OUTPUT_PATH = "D:\BandwidthDataset\cooked\\"
MILLISEC_IN_SEC = 1000.0

def plot_log_bandwidth():
	time_ms = []
	bytes_recv = []
	recv_time = []
	with open(LINK_FILE, 'r') as f:
		for line in f:
			parse = line.split()
			time_ms.append(float(parse[1]))
			bytes_recv.append(float(parse[-2]))
			recv_time.append(float(parse[-1]))
	time_ms = np.array(time_ms)
	bytes_recv = np.array(bytes_recv)
	recv_time = np.array(recv_time)
	throughput_all = bytes_recv / recv_time

	time_ms = time_ms - time_ms[0]
	time_ms = time_ms / MILLISECONDS_IN_SECONDS
	throughput_all = throughput_all * BITS_IN_BYTE / MBITS_IN_BITS * MILLISECONDS_IN_SECONDS

	plt.plot(time_ms, throughput_all)
	plt.xlabel('Time (second)')
	plt.ylabel('Throughput (Mbit/sec)')
	plt.show()

plot_log_bandwidth()
def calculate_network_tp():
    files = os.listdir(DATA_PATH)
    for f in files:
        file_path = DATA_PATH + f
        output_path = OUTPUT_PATH + f
        print(file_path)
        with open(file_path, 'r') as f, open(output_path, 'w') as mf:
            time = []
            mf.write('time_ms' + str("\t") + 'throughput_bpms' + '\n')
            for line in f:
                parse = line.split()
                if len(time) > 0 and float(parse[1]) < time[-1]:  # trace error, time not monotonically increasing
                    break
                time.append(float(parse[1]))
                time_ms = (float(parse[1]) - time[0])/MILLISEC_IN_SEC
                throughput_bpms = float(parse[-2]) / float(parse[-1]) * BITS_IN_BYTE / MBITS_IN_BITS * MILLISEC_IN_SEC
                mf.write(str(time_ms) + str("\t") + str(throughput_bpms) + '\n')


#calculate_network_tp()

current_dataframe = pd.read_csv(os.path.join(OUTPUT_PATH, 'report_bicycle_0001.log'), sep='\t')
print(current_dataframe.head())

def integration():
    dataframes = []
    files = os.listdir(OUTPUT_PATH)
    for f in files:
        file_path = OUTPUT_PATH + f
        print(file_path)
        current_dataframe = pd.read_csv(file_path, sep='\t')
        print(current_dataframe.head())
        dataframes.append(current_dataframe)
    dataframes = pd.concat(dataframes, axis=0, sort=True)
    print(dataframes.head())
    print('Number of data points: {0}'.format(dataframes.shape[0]))
    dataframes.to_csv(os.path.join(OUTPUT_PATH, 'bandwidth_modified.txt'), sep='\t', float_format='%.6f')


def generator_bandwidth(bandwidth_file_path):
    dataset = pd.read_csv(bandwidth_file_path, sep='\t')
    for i in range(len(dataset) ):
        yield dataset.loc[i, 'throughput_bpms']
        if i > len(dataset):
            raise StopIteration

gen_bandwidth = generator_bandwidth(os.path.join(OUTPUT_PATH, 'bandwidth_modified.txt'))
bandwidth = next(gen_bandwidth)


