#
# THIS IS THE ROBUSTMPC BUFFER-SCORE + WEIGHTED LINEAR REGRESSION (BS+WLR) VERSION
#
from typing import List
from collections import deque
import statistics
import itertools
import numpy as np
from sklearn import linear_model
from math import exp, log
import matplotlib.pyplot as plt

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
    global first_chunks_count, prev_throughputs, last_selected_index, prev_time, prev_times
    LOOK_AHEAD_SIZE = 5
    # For the first 5 chunks, we can't predict a throughput. Just pick the lowest quality.
    if first_chunks_count < MIN_PAST_VALUES:
        first_chunks_count += 1
        # Don't add the first throughput and time. It is zero.
        if first_chunks_count == 1:
            prev_time = client_message.total_seconds_elapsed
            return 0
        prev_throughputs.append(client_message.previous_throughput)
        prev_times.append(prev_time)
        prev_time = client_message.total_seconds_elapsed
        last_selected_index = 0
        return 0
    # Update the previous throughputs and times
    prev_throughputs.append(client_message.previous_throughput)
    prev_times.append(prev_time)
    prev_time = client_message.total_seconds_elapsed
    num_past_values = min(MAX_PAST_VALUES, len(prev_times))
    # Create the olslr model (call model.predict() to predict)
    # model = olslr_tp_model(np.array(prev_times[-num_past_values:]), np.array(prev_throughputs[-num_past_values:]))
    WEIGHTS = [1/9, 1/9, 2/9, 2/9, 3/9]
    model = wlslr_tp_model(np.array(prev_times[-num_past_values:]), np.array(prev_throughputs[-num_past_values:]), weights=np.array(WEIGHTS))
    # Combinations of possible choices of chunk qualities
    indices_list = list(range(client_message.quality_levels))
    # min taken here in case we are at the end of the simulation where we don't have as many upcoming quality bitrates
    current_and_upcoming_indices = [indices_list] * min(LOOK_AHEAD_SIZE, len(client_message.upcoming_quality_bitrates) + 1)
    combos = [p for p in itertools.product(*[client_message.quality_bitrates, *client_message.upcoming_quality_bitrates[0:LOOK_AHEAD_SIZE-1]])]
    combo_indices = [p for p in itertools.product(*current_and_upcoming_indices)]
    
    # Go through the different combinations and find the one that gives the highest score
    best_combo_index = 0
    best_combo_score = -100000
    for i, combo in enumerate(combos):
        combo_index_list = combo_indices[i]

        # Quality score is determined solely by the index chosen (quality 0 < quality 1 < quality 2 < ...)
        quality_score = statistics.mean(combo_index_list) * client_message.quality_coefficient
        # Variations are computed based on difference in quality indices chosen. Lowest -> Highest is higher variation than Middle -> Highest
        variation_score = calculate_variation(combo_index_list, last_selected_index) * client_message.variation_coefficient
        # Rebuffer score is based on the number of seconds of rebuffer
        predicted_times, predicted_throughputs = predict_throughputs(model, combo, client_message.total_seconds_elapsed, min(prev_throughputs), max(prev_throughputs))
        # if not np.all(np.isclose(prev_throughputs[-num_past_values:], prev_throughputs[-1])):
        #     print(prev_throughputs)
        #     plot_predictions(model, prev_times[-num_past_values:], prev_throughputs[-num_past_values:], predicted_times, predicted_throughputs)
        #     break
        # print(f"Predicted throughputs: {predicted_throughputs}")
        # print(f"Past throughputs: {prev_throughputs}")
        rebuffer_score = calculate_rebuffer_time(combo, predicted_throughputs, client_message.buffer_seconds_until_empty, client_message.buffer_seconds_per_chunk) * client_message.rebuffering_coefficient
        # Buffer size score determined using expontential. Multiplied with total score.
        buffer_score = calculate_buffer_score(combo, predicted_throughputs, client_message.buffer_seconds_until_empty, client_message.buffer_max_size, client_message.buffer_seconds_per_chunk)

        # total_score = quality_score - variation_score - rebuffer_score
        total_score = (quality_score - variation_score - rebuffer_score) * buffer_score
        # print(buffer_score)
        if total_score > best_combo_score:
            best_combo_index = i
            best_combo_score = total_score

    chosen_index = combo_indices[best_combo_index][0]
    last_selected_index = chosen_index
    return chosen_index

first_chunks_count = 0
# The minimum number of past throughputs before allowing predictions based on them
MIN_PAST_VALUES = 5
# The maximum number of past throughputs to use for predicting future throughputs
MAX_PAST_VALUES = 5
prev_throughputs = []
prev_times = []
prev_time = 0
last_selected_index = 0 

def calculate_buffer_score(chunk_sizes, predicted_throughputs, current_buffer, max_buffer_size, chunk_duration):
    """
    Calculate a score based on how full the buffer will be after downloading the future chunks given predicted throughputs.
    Score is between 0 to 1, following a curve of the form (1 - exp(-tau*B)), where B is final_predicted_buffer/max_buffer_size.
    """
    # Constant to determine how quickly the score ramps up. Below is set s.t. 30% buf -> score of 0.5.
    tau = -log(0.5)/0.3
    for i, chunk_size in enumerate(chunk_sizes):
        throughput = predicted_throughputs[i]
        download_time = chunk_size / throughput
        current_buffer = current_buffer - download_time + chunk_duration
        # If our current buffer goes negative, set it back to zero. Otherwise our final current buffer will not be accurate.
        if current_buffer < 0:
            current_buffer = 0
    return 1 - exp(-tau*current_buffer/max_buffer_size)

def calculate_variation(indices, start_index=None):
    """
    Compute the 'variation' of a list of indices. Sums the absolute values of the differences between entries in the list.
    """
    total_var = 0
    if start_index is None:
        prev_index = indices[0]
    else:
        prev_index = start_index
    for index in indices:
        total_var += abs(index - prev_index)
        prev_index = index
    return total_var

def calculate_rebuffer_time(chunk_sizes, predicted_throughputs, current_buffer, chunk_duration):
    """
    Compute how much rebuffer would take place given that we want to download the list of chunk sizes.
    It is assumed we can download at the rates specified in the predicted throughputs during each chunk.
    The current buffer size should be in seconds.
    The chunk duration is how many seconds of video is contained in each chunk (assumed to be the same for each chunk).
    
    Returns: The number of seconds of rebuffer that will occur.
    """
    total_rebuffer = 0
    for i, chunk_size in enumerate(chunk_sizes):
        throughput = predicted_throughputs[i]
        download_time = chunk_size / throughput
        # Check if a rebuffer will occur
        if current_buffer - download_time < 0:
            total_rebuffer += download_time - current_buffer
        current_buffer = current_buffer - download_time + chunk_duration
        # If our current buffer goes negative, set it back to zero. Otherwise total rebuffer time will be larger than it should be.
        if current_buffer < 0:
            current_buffer = 0
    return total_rebuffer

def max_error(iterable, mean):
    """
    Compute the maximum absolute percentage error of the iterable from the given mean
    """
    error_max = 0
    for item in iterable:
        error = abs((item - mean) / mean)
        if error > error_max:
            error_max = error
    return error_max

def olslr_tp_model(past_times, past_throughputs):
    """
    Use ordinary least squares linear regression to create a fitted model.

    Parameters
    ----------
    past_times : Previous times where throughputs were measured.
    past_throughputs : The throughputs corresponding to the previous times.

    Returns
    -------
    regr : The fitted model.
    """
    # Make sure shapes are correct
    past_times = past_times.reshape(len(past_times.flatten()), 1)
    past_throughputs = past_throughputs.reshape(len(past_throughputs.flatten()), 1)
    # Create the linear regression model
    regr = linear_model.LinearRegression()
    # Fit the linear regression model to the past data
    regr.fit(past_times, past_throughputs)
    # Return the fitted model
    return regr

def wlslr_tp_model(past_times, past_throughputs, weights):
    """
    Use weighted least squares linear regression to create a fitted model.

    Parameters
    ----------
    past_times : Previous times where throughputs were measured.
    past_throughputs : The throughputs corresponding to the previous times.
    weights : The weights to place on the different throughputs. The right-most weight is the most recent throughput.

    Returns
    -------
    regr : The fitted model.
    """
    # Make sure shapes are correct
    past_times = past_times.reshape(len(past_times.flatten()), 1)
    past_throughputs = past_throughputs.reshape(len(past_throughputs.flatten()), 1)
    # Create the linear regression model
    regr = linear_model.LinearRegression()
    # Fit the linear regression model to the past data
    regr.fit(past_times, past_throughputs, weights)
    # Return the fitted model
    return regr

def predict_throughputs(model, combo, current_time, min_pred=None, max_pred=None):
    """
    Predict future throughputs for a particular combo using a given model.

    Parameters
    ----------
    model : The fitted model to use to predict future throughputs.
    combo : The future chunk sizes to predict the throughputs for.
    current_time : The time to start predicting values for.

    Returns
    -------
    ts : The start-times corresponding to the predicted throughputs for the chunks in the combo.
    tps : The throughputs predicted for the chunks of combo.
    """
    tps = np.zeros([len(combo)])
    ts = np.zeros([len(combo)])
    for i, chunk_size in enumerate(combo):
        tp = model.predict([[current_time]])[0, 0]
        # tp cannot be negative, and ~0 values will make download time near infinite
        if min_pred is not None and tp < min_pred:
            tp = min_pred
        elif max_pred is not None and tp > max_pred:
            tp = max_pred
        tps[i] = tp
        ts[i] = current_time
        download_time = chunk_size / tp
        current_time += download_time
    return ts, tps

def plot_predictions(model, past_times, past_throughputs, future_times, future_throughputs):
    """
    Plot the past and future throughputs in a manner that displays the prediction clearly.

    Parameters
    ----------
    model : The model used to predict. Must have a predict() method.
    past_times : Previous times where throughputs were measured.
    past_throughputs : The throughputs corresponding to the previous times.
    future_times : The future times corresponding to the predicted throughputs.
    future_throughputs : The future throughputs.

    Returns
    -------
    Nothing. Just plots the data.
    """
    plt.figure()
    plt.scatter(past_times, past_throughputs)
    ys = model.predict([[past_times[0]]])[0, 0]
    ye = model.predict([[past_times[-1]]])[0, 0]
    # Plot the prediction line
    plt.plot([past_times[0], past_times[-1]], [ys, ye])
    plt.plot(future_times, future_throughputs, '.-')
    plt.legend(['Past Data', 'OLS Fit', 'Prediction'])
    plt.show()