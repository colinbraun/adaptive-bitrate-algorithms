from typing import List
from pprint import pprint

# Adapted from code by Zach Peats

# ======================================================================================================================
# Do not touch the client message class!
# ======================================================================================================================


class ClientMessage:
    """
    This class will be filled out and passed to student_entrypoint for your algorithm.
    """
    total_seconds_elapsed: float	  # The number of simulated seconds elapsed in this test
    previous_throughput: float		  # The measured throughput for the previous chunk in kB/s

    buffer_current_fill: float		    # The number of kB currently in the client buffer
    buffer_seconds_per_chunk: float     # Number of seconds that it takes the client to watch a chunk. Every
                                        # buffer_seconds_per_chunk, a chunk is consumed from the client buffer.
    buffer_seconds_until_empty: float   # The number of seconds of video left in the client buffer. A chunk must
                                        # be finished downloading before this time to avoid a rebuffer event.
    buffer_max_size: float              # The maximum size of the client buffer. If the client buffer is filled beyond
                                        # maximum, then download will be throttled until the buffer is no longer full

    # The quality bitrates are formatted as follows:
    #
    #   quality_levels is an integer reflecting the # of quality levels you may choose from.
    #
    #   quality_bitrates is a list of floats specifying the number of kilobytes the upcoming chunk is at each quality
    #   level. Quality level 2 always costs twice as much as quality level 1, quality level 3 is twice as big as 2, and
    #   so on.
    #       quality_bitrates[0] = kB cost for quality level 1
    #       quality_bitrates[1] = kB cost for quality level 2
    #       ...
    #
    #   upcoming_quality_bitrates is a list of quality_bitrates for future chunks. Each entry is a list of
    #   quality_bitrates that will be used for an upcoming chunk. Use this for algorithms that look forward multiple
    #   chunks in the future. Will shrink and eventually become empty as streaming approaches the end of the video.
    #       upcoming_quality_bitrates[0]: Will be used for quality_bitrates in the next student_entrypoint call
    #       upcoming_quality_bitrates[1]: Will be used for quality_bitrates in the student_entrypoint call after that
    #       ...
    #
    quality_levels: int
    quality_bitrates: List[float]
    upcoming_quality_bitrates: List[List[float]]

    # You may use these to tune your algorithm to each user case! Remember, you can and should change these in the
    # config files to simulate different clients!
    #
    #   User Quality of Experience =    (Average chunk quality) * (Quality Coefficient) +
    #                                   -(Number of changes in chunk quality) * (Variation Coefficient)
    #                                   -(Amount of time spent rebuffering) * (Rebuffering Coefficient)
    #
    #   *QoE is then divided by total number of chunks
    #
    quality_coefficient: float
    variation_coefficient: float
    rebuffering_coefficient: float
# ======================================================================================================================


# Your helper functions, variables, classes here. You may also write initialization routines to be called
# when this script is first imported and anything else you wish.


def student_entrypoint(client_message: ClientMessage):
    """
    Your mission, if you choose to accept it, is to build an algorithm for chunk bitrate selection that provides
    the best possible experience for users streaming from your service.

    Construct an algorithm below that selects a quality for a new chunk given the parameters in ClientMessage. Feel
    free to create any helper function, variables, or classes as you wish.

    Simulation does ~NOT~ run in real time. The code you write can be as slow and complicated as you wish without
    penalizing your results. Focus on picking good qualities!

    Also remember the config files are built for one particular client. You can (and should!) adjust the QoE metrics to
    see how it impacts the final user score. How do algorithms work with a client that really hates rebuffering? What
    about when the client doesn't care about variation? For what QoE coefficients does your algorithm work best, and
    for what coefficients does it fail?

    Args:
        client_message : ClientMessage holding the parameters for this chunk and current client state.

    :return: float Your quality choice. Must be one in the range [0 ... quality_levels - 1] inclusive.
    """
    global first, min_rate, max_rate, X, upper_reservoir
    # pprint(vars(client_message))
    if first:
        # print('first run')
        min_rate, max_rate = find_extremes(client_message.upcoming_quality_bitrates)
        print(f"Min rate: {min_rate}, Max rate: {max_rate}")
        first = False
        X = client_message.buffer_max_size * 2
        # Set the upper reservoir size to be at the 90% point (last 10% of buffer)
        upper_reservoir = 0.1 * client_message.buffer_max_size
    # print(client_message.previous_throughput)
    rate_map(client_message)
    return 0  # Let's see what happens if we select the lowest bitrate every time

first = True
last_selected_index = 0
min_rate = None
max_rate = None
X = None
upper_reservoir = None

def rate_map(client_message: ClientMessage):
    """
    Maps buffer occupancy to a rate from among the rates provided. Prefers maintaining the last_selected_index.
    """
    global last_selected_index, min_rate, max_rate, X, upper_reservoir
    bitrates = client_message.quality_bitrates
    occupancy = client_message.buffer_seconds_until_empty / client_message.buffer_max_size
    if client_message.previous_throughput == 0:
        return 0
    # print(X)
    reservoir = X - client_message.previous_throughput / bitrates[last_selected_index] * client_message.buffer_seconds_per_chunk * X
    # Make sure reservoir is a reasonable value
    if reservoir < 3:
        reservoir = 3
    elif reservoir > client_message.buffer_max_size/2:
        reservoir = client_message.buffer_max_size/2
    # The cushion is whatever remains of the buffer that isn't lower or upper reservoir
    cushion = client_message.buffer_max_size - reservoir - upper_reservoir
    print(f"Reservoir: {reservoir}")
    print(f"Occupancy: {occupancy}")
    r_max = bitrates[client_message.quality_levels - 1]
    r_min = bitrates[0]
    slope = (r_max - r_min) / (client_message.buffer_seconds_until_empty - reservoir)
    mapped_rate = (client_message.buffer_seconds_until_empty - reservoir) * slope
    prev_index = last_selected_index - 1 if last_selected_index != 0 else last_selected_index
    next_index = last_selected_index + 1 if last_selected_index != client_message.quality_levels - 1 else last_selected_index
    if client_message.buffer_seconds_until_empty < reservoir:
        # If our buffer is below the reservoir, choose the lowest quality.
        last_selected_index = 0
        return last_selected_index
    elif client_message.buffer_seconds_until_empty >= reservoir + cushion:
        last_selected_index = client_message.quality_levels - 1
        return last_selected_index
    # TODO: Handle the middle (linear) section of the map from occupancy to video rate
    elif next_index != last_selected_index and mapped_rate >= bitrates[next_index]:
        last_selected_index = next_index
        return last_selected_index
    else:
        # Otherwise just return the previous rate (don't change)
        return last_selected_index
    



def find_extremes(arr):
    """
    Find the minimum and maximum values in arr and return them. arr must be a list of lists
    """
    min_val = 30000000
    max_val = -30000000
    for sub_arr in arr:
        for item in sub_arr:
            if item < min_val:
                min_val = item
            if item > max_val:
                max_val = item
    
    return min_val, max_val