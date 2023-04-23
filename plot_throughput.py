import numpy as np
import matplotlib.pyplot as plt
import configparser
import re

config = configparser.ConfigParser()
trace = 'lo_avg_lo_var'
config.read(f'tests/{trace}.ini') 
throughputs = []
times = []
first = True
for key in config['throughput']:
    throughput_s = re.findall(r'([0-9]+\.*[0-9]*)', config['throughput'][key])[0]
    if not first:
        times.append(float(key))
        throughputs.append(prev_throughput)
    first = False
    times.append(float(key))
    throughput = float(throughput_s)
    throughputs.append(throughput)
    prev_throughput = throughput

plt.plot(times, throughputs, '.-')
plt.title(trace)
plt.xlabel('Time (s)')
plt.ylabel('Throughput (MB/s)')
# plt.show()
plt.savefig(f'{trace}.png')