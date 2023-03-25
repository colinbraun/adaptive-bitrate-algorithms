from typing import Tuple

class Scorecard:
    """
    A class for logging video player chunk choices and calculating the resulting view metrics
    """
    def __init__(self, quality_coeff: float, rebuffer_coeff: float, switch_coeff: float, chunk_length: float):
        """
        Args:
            quality_coeff : Used for calculating video QoE. See output_results for explanation.
            rebuffer_coeff : Used for calculating video QoE. See output_results for explanation.
            switch_coeff : Used for calculating video QoE. See output_results for explanation.
            chunk_length : # of seconds of video each chunk contains.
        """
        self.quality_coeff = quality_coeff
        self.rebuffer_coeff = rebuffer_coeff
        self.switch_coeff = switch_coeff
        self.chunk_length = chunk_length

        self.chunk_info = []
        self.rebuffers = []

    def log_bitrate_choice(self, time: float, quality: int, bitrate: float):
        """
        Logs one bitrate choice for the player
        Args:
            time : Time at which the chunk finishes downloading.
            quality : Quality level of the chunk.
            bitrate : # of megabytes the chunk takes up.
        """
        self.chunk_info.append(
            {'arrival time': time, 'quality': quality, 'bitrate': bitrate}
        )

    def log_rebuffer(self, time: float, rebuffer_length: float, chunknum: int):
        """
        Logs one rebuffer for the player
        Args:
            time : Time at which the rebuffer occurs.
            rebuffer_length : # of seconds the rebuffer lasts. If <= 0, no rebuffer is logged.
            chunknum : Which chunk is being waited on.
        """
        if rebuffer_length > .01:
            self.rebuffers.append(
                {'time': time, 'rebuffer_length': rebuffer_length, 'chunknum': chunknum}
            )

    def count_switches(self, print_output: bool = False) -> int:
        """
        Counts the number of quality switches that have occurred since logging began.
        Args:
            print_output : Whether to print switch info.
        :return: int Total variation
        """
        variation = 0
        text = ''
        for i in range(1, len(self.chunk_info)):
            if self.chunk_info[i]['quality'] == self.chunk_info[i - 1]['quality']:
                continue

            # Found a switch!
            variation += abs(self.chunk_info[i]['quality'] - self.chunk_info[i - 1]['quality'])
            if print_output:
                text += f'\tQuality switch detected!.' \
                        f' Chunk {i - 1} quality {self.chunk_info[i - 1]["quality"]} ->' \
                        f' Chunk {i} quality {self.chunk_info[i]["quality"]}.' \
                        f' Changed by {abs(self.chunk_info[i - 1]["quality"] - self.chunk_info[i]["quality"])}.\n'

        if print_output:
            print(f'{variation} total variation detected.\n')
            print(text)
        return variation

    def get_rebuffer_time(self, print_output: bool = False) -> float:
        """
        Calculates the total amount of rebuffering that occurred since logging began.
        Args:
            print_output : Whether to print rebuffering info.
        :return: float total rebuffer time
        """
        rebuff_time = sum(r['rebuffer_length'] for r in self.rebuffers)
        text = ''
        for rebuffer in self.rebuffers:
            if print_output:
                text += f'\tRebuffer at time {rebuffer["time"]:.2f} detected! ' \
                        f'Lasted {rebuffer["rebuffer_length"]:.2f}' \
                        f' seconds. Buffering between chunks {rebuffer["chunknum"] - 1} and {rebuffer["chunknum"]}\n'

        if print_output:
            print(f'{len(self.rebuffers)} rebuffers detected. Total rebuffer time: {rebuff_time:.2f}')
            print(text)

        return rebuff_time

    def get_total_quality(self, print_output: bool = False) -> int:
        """
        Calculates the aggregate video quality since logging began.
        Args:
            print_output : Whether to print quality info.
        :return: float total video quality
        """
        total = sum(c['quality'] for c in self.chunk_info)
        if print_output:
            print(f'Total chunk quality is {total}, average chunk quality {round(total / len(self.chunk_info), 3)}\n')
        return total

    def output_results(self, verbose: bool = False) -> float:
        """
        Prints out the results for this playback. Includes switch, rebuffer, and quality info.
        Args:
            verbose : Whether to print in-depth info on each component.
        :return: float calculated user quality of experience
        """
        print('=' * 120)
        print('Test results:\n')
        if verbose:
            print('Chunk overview:')
            for i, c in enumerate(self.chunk_info):
                print(f'\tChunk {i} finished downloading at time {c["arrival time"]:.2f}.'
                      f' Quality {c["quality"]}, chunk size {c["bitrate"]:.2f}.')

            print('\n')
        total_quality = self.get_total_quality(print_output=verbose)
        rebuff_time = self.get_rebuffer_time(print_output=verbose)
        variation = self.count_switches(print_output=verbose)

        print('Test results:')
        print(f'\tTotal quality:            {total_quality:.2f}')
        print(f'\tTotal rebuffer time:      {rebuff_time:.2f}')
        print(f'\tTotal variation:          {variation:.2f}')
        print(f'User quality of experience = '
              f'[{self.quality_coeff:.2f}(Quality)'
              f' - {self.rebuffer_coeff:.2f}(Rebuffer Time)'
              f' - {self.switch_coeff:.2f}(Variation)] / (Chunk Count)')

        qoe = total_quality * self.quality_coeff - rebuff_time * self.rebuffer_coeff - variation * self.switch_coeff
        qoe /= len(self.chunk_info)
        print(f'User quality of experience: {qoe:.3f}\n')
        print('=' * 120)

        return qoe

    def get_qual_rebuff_var_qoe(self) -> Tuple[float, float, float, float]:
        """
        Returns the results for this test without printing anything
        :return: Tuple with the total quality, rebuffer time, total variation, and user QoE
        """
        total_quality = self.get_total_quality()
        rebuff_time = self.get_rebuffer_time()
        variation = self.count_switches()
        qoe = total_quality * self.quality_coeff - rebuff_time * self.rebuffer_coeff - variation * self.switch_coeff
        qoe /= len(self.chunk_info)

        return total_quality, variation, rebuff_time, qoe
