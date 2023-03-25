import os
import simulator
from importlib import reload
import sys

TEST_DIRECTORY = './tests'


def main(student_algo: str):
    """
    Runs simulator and student algorithm on all tests in TEST_DIRECTORY
    Args:
        student_algo : Student algorithm to run
    """
    # Run main loop, print output
    sum_qoe = 0
    print(f'\nTesting student algorithm {student_algo}')
    for test in os.listdir(TEST_DIRECTORY):
        reload(simulator)
        quality, variation, rebuff, qoe = simulator.main(os.path.join(TEST_DIRECTORY, test), student_algo, False, False)
        print(f'\tTest {test: <12}:'
              f' Total Quality {quality:8.2f},'
              f' Total Variation {variation:8.2f},'
              f' Rebuffer Time {rebuff:8.2f},'
              f' Total QoE {qoe:8.2f}')
        sum_qoe += qoe

    print(f'\n\tAverage QoE over all tests: {sum_qoe / len(os.listdir(TEST_DIRECTORY)):.2f}')


if __name__ == "__main__":
    assert len(sys.argv) >= 2, f'Proper usage: python3 {sys.argv[0]} [student_algo]'
    if sys.argv[1] != 'RUN_ALL':
        main(sys.argv[1])
    else:
        for algo in os.listdir('./student'):
            if algo[:len('student')] != 'student':
                continue
            name = algo[len('student'):].split('.')[0]
            main(name)