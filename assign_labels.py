import numpy as np


def assign_labels(entity_data, levels, channel_names):
    for entity_info in entity_data:
        num_intervals = len(entity_info['intervals'])
        for i in range(num_intervals):
            interval_data = entity_info['intervals'][i]
            for band_name, band_info in interval_data['bands'].items():
                for channel_idx, channel_name in enumerate(channel_names):
                    mean_power = band_info[channel_idx]['mean_power']

                    # Find the closest level based on mean power
                    closest_level = min(levels[band_name][channel_idx]['level_values'],
                                        key=lambda x: abs(mean_power - x))

                    # Get the number of the closest level
                    level_number = np.where(levels[band_name][channel_idx]['level_values'] == closest_level)

                    # Assign the label with the desired structure
                    label = f"{channel_name}-{band_name}-L{level_number[0][0]}"

                    # Assign the label to the corresponding channel in the entity info
                    interval_data['bands'][band_name][channel_idx]['label'] = label

    return entity_data
