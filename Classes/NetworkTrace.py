import math
from typing import List, Tuple

class NetworkTrace:
    """
    Class to hold a list of network bandwidths and simulate download times
    """
    def __init__(self, bandwidths: List[Tuple[float, float]]):
        """
        Args:
            bandwidths : List of tuples, (Start time in seconds, bandwidth in Mbps)
        """
        self.bwlist = bandwidths

    def get_current_timesegment(self, cur_time: float) -> Tuple[float, float]:
        """ Returns the time segement of cur_time as a tuple (Start time in seconds, bandwidth in Mbps) """
        return min(self.bwlist, key=lambda x: abs(x[0] - cur_time) if cur_time > x[0] else math.inf)

    def simulate_download_from_time(self, time: float, size: float) -> float:
        """
        Calculates the amount of time it takes for a chunk of the given size to be downloaded starting from the given
        time under the varying bandwidths of this trace
        Args:
            time : Download start time (seconds)
            size : Size of the download in Mb
        :return: float Number of seconds to download
        """
        cum_time = 0
        timeseg = self.get_current_timesegment(time)
        while True:
            # Find next bandwidth change
            try:
                next_set = self.bwlist[self.bwlist.index(timeseg) + 1]
            except IndexError:
                cum_time += size / timeseg[1]
                return cum_time

            # Drain download by time and throughput
            down_time = next_set[0] - time
            cum_time += down_time
            size -= down_time * timeseg[1]

            # Refund unused time
            if size <= 0:
                cum_time += size / timeseg[1]
                return cum_time

            timeseg = next_set
            time = timeseg[0]