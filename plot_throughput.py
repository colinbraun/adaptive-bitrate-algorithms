import numpy as np
import matplotlib.pyplot as plt
import configparser
import re

config = configparser.ConfigParser()
config.read('tests/hi_avg_lo_var.ini') 
throughputs = []
times = []
print(config['throughput'])
for key in config['throughput']:
    print(key)
    # print(config['throughput'][str(i)])
    print(config['throughput'][key])
    # time_s, throughput_s = re.findall(r'([0-9]+)\s*=\s*([0-9]+\.*[0-9]*)', config['throughput'][key])
    throughput_s = re.findall(r'([0-9]+\.*[0-9]*)', config['throughput'][key])[0]
    print(throughput_s)
    times.append(float(key))
    throughputs.append(float(throughput_s))

print(len(times))
plt.plot(times, throughputs, '.-')
plt.show()