import json
import argparse
import traceback
from load_entity_data import *
from calculate_intervals import *
from allocate_data import *
from interpolate_data_and_calculate_mean_power import *
from assign_labels import *
from save_abstracted_features import *
from save_and_load_workspace import *


def read_config(config_file_path):
    with open(config_file_path, "r") as f:
        config = json.load(f)
    return config


def main(config_file_path):
    try:

        config = read_config(config_file_path)

        sample_rate = config["sample_rate"]
        mats_folder_path = config["mats_folder_path"]
        interval_length = config["interval_length_ms"]
        number_of_categories = config["number_of_categories"]
        frequency_bands = config["frequency_bands"]
        output_folder = config["output_folder"]
        direct_to_step = config["direct_to_step"]
        levels_to_save = config["levels_to_save"]

        print("Sample Rate:", sample_rate)
        print("MATs Folder Path:", mats_folder_path)
        print("Interval Length in mS:", interval_length)
        print("Number of Categories:", number_of_categories)
        print("Frequency Bands", frequency_bands)
        print("Output Folder:", output_folder)
        print("Direct To Step:", direct_to_step)
        print("Levels To Save:", levels_to_save)

        if direct_to_step != 'True':

            # Step1: Load entity data
            entity_data, channel_names = load_entity_data(config['mats_folder_path'])
            print("Finished Step 1: Load entity data")

            # Step 2: Calculate intervals
            calculate_intervals(entity_data, interval_length, sample_rate)
            print("Finished Step 2: Calculate intervals")

            # Step 3: Allocate data
            allocate_data(entity_data, sample_rate, interval_length)
            print("Finished Step 3: Allocate data")

            # Step 4: Calculate mean power
            levels = interpolate_data_and_calculate_mean_power(
                entity_data, frequency_bands, sample_rate, number_of_categories, interval_length)
            print("Finished Step 4: Interpolate and Calculate mean power")

        else:
            entity_data = load_workspace('entity_data.pkl')
            channel_names = load_workspace('channel_names.pkl')
            levels = load_workspace('levels.pkl')

        # Step 5: Assign labels
        entity_data = assign_labels(entity_data, levels, channel_names)
        print("Finished Step 5: Assign labels")

        # Step 6: Save abstracted features
        save_abstracted_features(entity_data, channel_names, frequency_bands,
                                 number_of_categories, output_folder, levels_to_save)
        print("Finished Step 6: Save abstracted features")
        print("Great Success!!!")

    except Exception as e:
        # Print the error message
        print("An error occurred:", str(e))

        # Print the traceback
        traceback.print_exc()

        # Save the workspace

        save_workspace(entity_data, 'entity_data.pkl')
        save_workspace(channel_names, 'channel_names.pkl')
        save_workspace(levels, 'levels.pkl')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script Configuration")
    parser.add_argument("--config", type=str, help="Path to the configuration file")

    args = parser.parse_args()

    if args.config:
        main(args.config)
    else:
        print("Please specify a configuration file using --config argument.")



