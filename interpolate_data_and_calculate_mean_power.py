import scipy.stats as stats
from scipy.integrate import simps
from scipy import signal
import numpy as np


def calculate_power(data, sample_rate):
    win = len(data)
    freq, psd = signal.welch(data, sample_rate, nperseg=win)

    return freq, psd


def interpolate_data_and_calculate_mean_power(entity_data, frequency_bands, sample_rate, number_of_categories, interval_length):

    num_channels = np.size(entity_data[0]['intervals'][0]['data'], 0)
    levels = {band_name: {channel: {}
                          for channel in range(num_channels)}
              for band_name in frequency_bands}
    mean_powers = {}
    # Initialize the mean powers for each band and channel
    for band in frequency_bands:
        mean_powers[band] = {}
        for channel in range(num_channels):
            mean_powers[band][channel] = []

    for entity_info in entity_data:
        num_intervals = len(entity_info['intervals'])
        for i in range(num_intervals):
            allocated_data = entity_info['intervals'][i]['data']
            num_channels = allocated_data.shape[0]
            num_samples = allocated_data.shape[1]
            for channel in range(num_channels):
                channel_data = np.append(allocated_data[channel], np.zeros(sample_rate-num_samples))
                for band_name, [low_freq, high_freq] in frequency_bands.items():

                    # Calculate the power
                    freq, psd = calculate_power(channel_data, sample_rate)

                    # Extract the power within the frequency band of interest
                    band_mask = (freq >= low_freq) & (freq <= high_freq)

                    # Frequency resolution
                    freq_res = freq[1] - freq[0]

                    # Compute the absolute power by approximating the area under the curve
                    mean_power = simps(psd[band_mask], dx=freq_res)
                    mean_powers[band_name][channel].append(mean_power)

                    # Check if the key exists in the bands dictionary and create it if not
                    if band_name not in entity_info['intervals'][i]['bands']:
                        entity_info['intervals'][i]['bands'][band_name] = {}
                    entity_info['intervals'][i]['bands'][band_name][channel] = {'mean_power': mean_power}
            entity_info['intervals'][i]['data'] = []

    # Calculate levels based on mean power values across all entities
    for band_name in frequency_bands:

        # Calculate power for each channel within the frequency band
        for channel in range(num_channels):
            mean_power_avg = np.mean(mean_powers[band_name][channel])
            mean_power_std = np.std(mean_powers[band_name][channel])

            # Calculate percentiles based on normal distribution
            percentiles = np.linspace(0, 100, number_of_categories+2)[1:-1]
            level_values = stats.norm.ppf(percentiles / 100, loc=mean_power_avg, scale=mean_power_std)

            levels[band_name][channel]['level_values'] = level_values

    return levels
