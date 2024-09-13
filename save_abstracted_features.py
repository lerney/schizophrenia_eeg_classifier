import itertools
import numpy as np
import json

def generate_abstracted_features(channel_names, frequency_bands, levels_to_save):
    abstracted_features = []

    # Generate combinations of channel names, frequency bands, and number of categories
    combinations = list(itertools.product(channel_names, frequency_bands.keys(), np.array(levels_to_save)))

    # Format the combinations as abstracted features
    for channel, band, level in combinations:
        feature = f"{channel}-{band}-L{level}"
        abstracted_features.append(feature)

    return abstracted_features


def save_abstracted_features(entity_data, channel_names, frequency_bands,
                             number_of_categories, output_folder, levels_to_save):
    abstracted_features = generate_abstracted_features(channel_names, frequency_bands, levels_to_save)

    for entity_info in entity_data:
        entity_data_records = {"entityName": entity_info['entity_name'],"class": 1,"records": []}

        for interval_info in entity_info['intervals']:
            start_time = interval_info['startTime']
            end_time = interval_info['endTime']
            for band_name, band_info in interval_info['bands'].items():
                for channel_idx, channel_name in enumerate(channel_names):
                    label = interval_info['bands'][band_name][channel_idx]['label']
                    if int(label[-1]) in np.array(levels_to_save):
                        # Create a record with start time, end time, and label
                        record = {"startTime": start_time,"endTime": end_time,"label": label}

                        # Add the record to the entity's data
                        entity_data_records['records'].append(record)

        # Generate the output file name
        output_file = f"{output_folder}\\{entity_info['entity_name']}_freq_power_abs.json"

        # Create the final JSON object
        output_data = {
            "abstractedFeatures": abstracted_features,
            "entitiesData": [entity_data_records]
        }

        # Save the output JSON to a file
        with open(output_file, 'w') as f:
            json.dump(output_data, f)
