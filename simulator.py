import configparser
import importlib
from typing import Tuple, List, Type
from Classes import SimBuffer, NetworkTrace, Scorecard
import sys
from importlib import reload
import os

# ======================================================================================================================
# CONFIG PARAMETERS
# ======================================================================================================================
VIDEO_HEADING	   = 'video'
CHUNK_LENGTH		= 'chunk_length'
CLIENT_BUFF_SIZE	= 'client_buffer_size'

QUALITY_HEADING	 = 'quality'
QUALITY_LEVELS	  = 'quality_levels'
BASE_CHUNK_SIZE = 'base_chunk_size'
QUAL_COEF		   = 'quality_coefficient'
BUF_COEF			= 'rebuffering_coefficient'
SWITCH_COEF		 = 'variation_coefficient'

THROUGHPUT_HEADING  = 'throughput'

CHUNK_SIZE_RATIOS_HEADING  = 'chunk_size_ratios'
CHUNK_SIZE_RATIOS		 = 'chunk_size_ratios'


def read_test(config_path: str, print_output: bool):
	"""
	Reads and loads parameters from config_path
	Args:
		config_path : .ini file to read
		print_output : Whether to print output
	:return:
		Tuple containing the NetworkTrace, Scorecard, SimBuffer, a list of chunk quality bitrates,
		and the chunk duration. The chunk quality options are formatted as a list of lists. e.g.
		chunk_qualities[3][1] = number of bytes for chunk index 3, quality index 1.
	"""
	try:
		if print_output: print(f'\nLoading test file {config_path}.')
		cfg = configparser.RawConfigParser(allow_no_value=True, inline_comment_prefixes='#')
		cfg.read(config_path)

		chunk_length = float(cfg.get(VIDEO_HEADING, CHUNK_LENGTH))
		base_chunk_cost = float(cfg.get(VIDEO_HEADING, BASE_CHUNK_SIZE))
		client_buffer_size = float(cfg.get(VIDEO_HEADING, CLIENT_BUFF_SIZE))
		if print_output: print(f'\tLoaded chunk length {chunk_length} seconds, base cost {base_chunk_cost} megabytes.')

		quality_levels = int(cfg.get(QUALITY_HEADING, QUALITY_LEVELS))
		if print_output: print(f'\tLoaded {quality_levels} quality levels available.')

		quality_coefficient = float(cfg.get(QUALITY_HEADING, QUAL_COEF))
		rebuffering_coefficient = float(cfg.get(QUALITY_HEADING, BUF_COEF))
		variation_coefficient = float(cfg.get(QUALITY_HEADING, SWITCH_COEF))
		if print_output: print(f'\tLoaded {quality_coefficient} quality coefficient,'
							   f' {rebuffering_coefficient} rebuffering coefficient,'
							   f' {variation_coefficient} variation coefficient.')

		throughputs = dict(cfg.items(THROUGHPUT_HEADING))
		throughputs = [(float(time), float(throughput)) for time, throughput in throughputs.items()]
		if print_output: print(f'\tLoaded {len(throughputs)} different throughputs.')

		chunks = cfg.get(CHUNK_SIZE_RATIOS_HEADING, CHUNK_SIZE_RATIOS)
		chunks = list(float(x) for x in chunks.split(',') if x.strip())
		chunk_qualities = [[c * (2**i) * base_chunk_cost for i in range(quality_levels)] for c in chunks]
		if print_output: print(f'\tLoaded {len(chunks)} chunks. Total video length is {len(chunks) * chunk_length} seconds.')

		trace = NetworkTrace.NetworkTrace(throughputs)
		logger = Scorecard.Scorecard(quality_coefficient, rebuffering_coefficient, variation_coefficient, chunk_length)
		buffer = SimBuffer.SimBuffer(chunk_length, client_buffer_size)

		if print_output: print(f'\tDone reading config!\n')

		return trace, logger, buffer, chunk_qualities, chunk_length

	except:
		print('Exception reading config file!')
		import traceback
		traceback.print_exc()
		exit()


# ======================================================================================================================
# MAIN
# ======================================================================================================================
def main(config_file: str, student_algo, verbose: bool, print_output=True) -> Tuple[float, float, float, float]:
	"""
	Main loop. Runs the simulator with the given config file.
	Args:
		config_file : Path to the config file of this test
		student_algo: Student algorithm to run
		verbose : Whether to print verbose output
		print_output : Whether to print any output at all
	:return: Tuple with the total quality, rebuffer time, total variation, and user QoE for this test
	"""
	trace, logger, buffer, chunk_qualities, chunk_length = read_test(config_file, print_output)

	assert os.path.exists(f'./student/student{student_algo}.py'),\
		f'Could not find student algorithm ./student/student{student_algo}.py!'
	student = importlib.import_module(f'student.student{student_algo}')
	reload(student)  # In case the student has global variables

	current_time = 0
	prev_throughput = 0

	# Communication loop with student (for all chunks):
	for chunknum in range(len(chunk_qualities)):
		# Set up message for student
		message = student.ClientMessage()
		message.total_seconds_elapsed = current_time
		message.previous_throughput = prev_throughput
		# Buffer
		message.buffer_seconds_per_chunk = chunk_length
		message.buffer_seconds_until_empty = buffer.seconds_left
		message.buffer_max_size = buffer.client_buffer_size

		# Video
		message.quality_levels = len(chunk_qualities[chunknum])
		message.quality_bitrates = chunk_qualities[chunknum]
		message.upcoming_quality_bitrates = chunk_qualities[chunknum+1:] if chunknum < len(chunk_qualities) - 1 else []
		# Quality
		message.quality_coefficient = logger.quality_coeff
		message.rebuffering_coefficient = logger.rebuffer_coeff
		message.variation_coefficient = logger.switch_coeff

		# Call student algorithm
		quality = student.student_entrypoint(message)
		if quality < 0 or quality >= len(chunk_qualities[chunknum]) or not isinstance(quality, int):
			print("Student returned invalid quality, exiting")
			break
		chosen_bitrate = chunk_qualities[chunknum][quality]

		# Simulate download
		time_elapsed = trace.simulate_download_from_time(current_time, chosen_bitrate)
		rebuff_time = buffer.sim_chunk_download(chosen_bitrate, time_elapsed)

		# Update state variables and log
		prev_throughput = chosen_bitrate / time_elapsed
		current_time += time_elapsed
		current_time += buffer.wait_until_buffer_is_not_full(verbose and print_output)
		logger.log_bitrate_choice(current_time, quality, chosen_bitrate)
		logger.log_rebuffer(current_time - rebuff_time, rebuff_time, chunknum)

	if print_output:
		logger.output_results(verbose=verbose)

	return logger.get_qual_rebuff_var_qoe()


if __name__ == '__main__':
	assert len(sys.argv) >= 3, f'Proper usage: python3 {sys.argv[0]} [config_file] [student_algo] [-v --verbose]'
	main(sys.argv[1], sys.argv[2], '-v' in sys.argv or '--verbose' in sys.argv)
