
class SimBuffer:
    """
    A class to hold and simulate a buffer
    """
    def __init__(self, chunk_duration: float, client_buffer_size: float):
        """
        Args:
            chunk_duration : Number of seconds of video each cunk carries.
            client_buffer_size : Number of seconds of video the client can hold.
        """
        self.chunk_duration = chunk_duration
        self.client_buffer_size = client_buffer_size

        self.seconds_left = 0
        self.chunks = []
        self.seconds_played = 0

    def get_occupancy(self) -> float:
        """ Returns #Mb in the buffer. """
        current_chunk = int(self.seconds_played // self.chunk_duration + .001)  # .001 in case of rounding errors
        if current_chunk < len(self.chunks):
            return sum(self.chunks[current_chunk:])
        return 0

    def sim_chunk_download(self, chunk_size: float, playback_time) -> float:
        """
        Simulates a chunk download. Adds this chunk to the buffer and reduces the buffer by enough chunks to play
        playback_time. Returns number of seconds rebuffered, or 0 if no rebuffer.
        Args:
            chunk_size : Size of incoming chunk in Mb.
            playback_time : Number of seconds to remove from buffer

        :return: float Number of seconds rebuffered
        """
        rebuffer_time = self.burn_time(playback_time)
        self.chunks.append(chunk_size)
        self.seconds_left += self.chunk_duration
        return rebuffer_time

    def burn_time(self, playback_time: float) -> float:
        """
        Reduces the buffer by enough chunks to play for playback_time. Returns number of seconds rebuffered,
        or 0 if no rebuffer.
        Args:
            playback_time : Number of seconds to remove from buffer

        :return: float Number of seconds rebuffered
        """
        rebuffer_time = max(playback_time - self.seconds_left, 0)
        self.seconds_left = max(self.seconds_left - playback_time, 0)
        self.seconds_played += min(playback_time, self.seconds_left)
        return rebuffer_time

    def wait_until_buffer_is_not_full(self, verbose: bool) -> float:
        """
        Returns the amount of time until the buffer is not full. Burns the same amount of time to reduce the buffer
        to max occupancy.
        Args:
            verbose : Boolean whether to print output or not
        :return: float Number of seconds to wait
        """
        wait_time = max(self.seconds_left - self.client_buffer_size, 0)
        self.burn_time(wait_time)
        if verbose and wait_time:
            print(f'Buffer full! Waiting {wait_time:.2f} seconds before downloading another chunk.')
        return wait_time
