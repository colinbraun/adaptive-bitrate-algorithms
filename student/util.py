import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from math import exp, log
import matplotlib.pyplot as plt

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
