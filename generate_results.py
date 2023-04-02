import numpy as np
import subprocess
import re
import sys

TEST_CASES = ['hi_avg_hi_var', 'hi_avg_mi_var', 'hi_avg_lo_var', 'mi_avg_hi_var', 'mi_avg_mi_var', 'mi_avg_lo_var', 'lo_avg_hi_var', 'lo_avg_mi_var', 'lo_avg_lo_var']
TEST_CASES = reversed(TEST_CASES)
for test_case in TEST_CASES:
    # cmd = f'python3 simulator.py tests/{test_file}'
    # p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    out = re.findall(r'User quality of experience: (-?[0-9]+\.[0-9]*)', subprocess.check_output(["python3", "simulator.py", f"tests/{test_case}.ini", sys.argv[1]]).decode("utf-8"))[0]
    # print(type(out))
    print(f"{test_case}: {out}")

