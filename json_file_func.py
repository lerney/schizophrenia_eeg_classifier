import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def time_to_seconds(time_str):
    hours, minutes, seconds = map(float, time_str.split(':'))
    total_seconds = 3600*hours + 60*minutes + seconds
    return round(total_seconds, 3)


def filter_json_sec_folders(input_folder, output_folder, sec_beginning, sec_middle, sec_end):
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            input_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename)
            filter_json_sec(input_file_path, output_file_path, sec_beginning, sec_middle, sec_end)


def filter_json_sec(input_file_path, output_file_path, sec_beginning, sec_middle, sec_end):
    with open(input_file_path, 'r') as f:
        data = json.load(f)

    for entity in data['entitiesData']:
        records = entity['records']
        if len(records) > 0:
            filtered_records = []

            # Filter records from the beginning
            start_time_beginning = datetime.strptime(records[0]['startTime'], '%H:%M:%S.%f') + timedelta(seconds=5)
            end_time_beginning = start_time_beginning + timedelta(seconds=sec_beginning)
            for record in records:
                record_start_time = datetime.strptime(record['startTime'], '%H:%M:%S.%f')
                if start_time_beginning <= record_start_time <= end_time_beginning:
                    filtered_records.append(record)

            # Filter records from the middle
            start_time_middle = datetime.strptime(records[0]['startTime'], '%H:%M:%S.%f') + timedelta(seconds=100)
            end_time_middle = start_time_middle + timedelta(seconds=sec_middle)
            for record in records:
                record_start_time = datetime.strptime(record['startTime'], '%H:%M:%S.%f')
                if start_time_middle <= record_start_time <= end_time_middle:
                    filtered_records.append(record)

            # Filter records from the end
            start_time_end = datetime.strptime(records[-1]['startTime'], '%H:%M:%S.%f') - timedelta(seconds=sec_end)
            end_time_end = datetime.strptime(records[-1]['startTime'], '%H:%M:%S.%f')
            for record in records:
                record_start_time = datetime.strptime(record['startTime'], '%H:%M:%S.%f')
                if start_time_end <= record_start_time <= end_time_end:
                    filtered_records.append(record)

            entity['records'] = filtered_records

    file_name, extension = os.path.splitext(output_file_path)
    new_file_name = file_name + "_" + extension

    with open(new_file_name, 'w') as f:
        json.dump(data, f, indent=4)


# function to merge json files
def merge_json_files(input_folder, output_file):
    data = defaultdict(list)
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            input_file = os.path.join(input_folder, filename)
            with open(input_file) as f:
                json_data = json.load(f)
                for feature in json_data['abstractedFeatures']:
                    if feature not in data['abstractedFeatures']:
                        data['abstractedFeatures'].append(feature)
                for entity in json_data['entitiesData']:
                    if entity['entityName'] not in [d['entityName'] for d in data['entitiesData']]:
                        data['entitiesData'].append(entity)
                    else:
                        for d in data['entitiesData']:
                            if d['entityName'] == entity['entityName']:
                                d['records'] += entity['records']
                                break
    with open(output_file, 'w') as f:
        json.dump(data, f)


def divide_entities_and_merge(input_folder, output_file, seconds, overlap):
    # create new entities and merged json file for training entities (kind of bootstrapping)
    # Get all JSON files in the input folder
    json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]

    # Initialize merged JSON data
    merged_json_data = {"abstractedFeatures": [], "entitiesData": []}

    # Loop through all JSON files
    for json_file in json_files:
        with open(os.path.join(input_folder, json_file), 'r') as f:
            json_data = json.load(f)

        # Add abstracted features to merged data (avoiding duplicates)
        for feature in json_data["abstractedFeatures"]:
            if feature not in merged_json_data["abstractedFeatures"]:
                merged_json_data["abstractedFeatures"].append(feature)

        # Loop through all entities in JSON data
        for entity in json_data["entitiesData"]:
            entity_name = entity["entityName"]

            # Get start and end times for the entity
            start_time = time_to_seconds(entity["records"][0]["startTime"])
            end_time = time_to_seconds(entity["records"][-1]["endTime"])

            # Filter records based on specified time intervals
            time_intervals = []
            num_intervals = int((end_time - start_time - (seconds-overlap)) / overlap)
            for i in range(num_intervals):
                interval_start_time = start_time + i*overlap
                interval_end_time = interval_start_time + seconds
                filtered_records = [record for record in entity["records"] if
                                    interval_start_time <= time_to_seconds(record["startTime"]) <= interval_end_time
                                    and interval_start_time <= time_to_seconds(record["endTime"]) <= interval_end_time]

            # Add filtered records to merged data (with new entity names)
                new_entity_name = f"{entity_name}-{str(i+1).zfill(2)}"
                new_entity_data = {"entityName": new_entity_name, "class": entity["class"], "records": filtered_records}
                merged_json_data["entitiesData"].append(new_entity_data)

    # Write merged JSON data to output file
    with open(output_file, 'w') as f:
        json.dump(merged_json_data, f, indent=4)


def divide_entities_without_merge(input_folder, output_folder, seconds, overlap):
    # create new entities and individual JSON files for training entities
    # Get all JSON files in the input folder
    json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]

    # Loop through all JSON files
    for json_file in json_files:
        with open(os.path.join(input_folder, json_file), 'r') as f:
            json_data = json.load(f)

        # Initialize filtered JSON data for the current file
        filtered_json_data = {"abstractedFeatures": json_data["abstractedFeatures"], "entitiesData": []}

        # Loop through all entities in JSON data
        for entity in json_data["entitiesData"]:
            entity_name = entity["entityName"]
            entity_class = entity["class"]

            # Get start and end times for the entity
            start_time = time_to_seconds(entity["records"][0]["startTime"])
            end_time = time_to_seconds(entity["records"][-1]["endTime"])

            # Filter records based on specified time intervals
            time_intervals = []
            num_intervals = int((end_time - start_time - (seconds - overlap)) / overlap)
            for i in range(num_intervals):
                interval_start_time = start_time + i * overlap
                interval_end_time = interval_start_time + seconds
                filtered_records = [record for record in entity["records"] if
                                    interval_start_time <= time_to_seconds(record["startTime"]) <= interval_end_time
                                    and interval_start_time <= time_to_seconds(record["endTime"]) <= interval_end_time]

                # Add filtered records to filtered JSON data (with new entity names)
                new_entity_name = f"{entity_name}-{str(i+1).zfill(2)}"
                new_entity_data = {"entityName": new_entity_name, "class": entity_class, "records": filtered_records}
                filtered_json_data["entitiesData"].append(new_entity_data)

                # Write filtered JSON data to individual output files
                file_name, extension = os.path.splitext(json_file)
                new_file_name = f"{new_entity_name}.json"
                new_file_path = os.path.join(output_folder, new_file_name)
                with open(new_file_path, 'w') as f:
                    json.dump(filtered_json_data, f, indent=4)
                filtered_json_data = {"abstractedFeatures": json_data["abstractedFeatures"], "entitiesData": []}


def train_test_split_by_subject(df, label, subject, test_size=0.3, random_state=42):
    # Get unique subjects
    subjects = df[subject].unique()

    # Get original label ratio
    original_df_label_ratio = df[label].value_counts(normalize=True)

    # Initialize best ratio and dataframes
    best_ratio_diff = np.inf
    best_train_df = None
    best_test_df = None

    # Loop over random states
    for i in range(15):
        # Shuffle subjects
        np.random.seed(random_state + i)
        np.random.shuffle(subjects)

        # Split subjects into train and test
        n_test_subjects = int(test_size * len(subjects))
        test_subjects = subjects[:n_test_subjects]
        train_subjects = subjects[n_test_subjects:]

        # Split data by train and test subjects
        train_df = df[df[subject].isin(train_subjects)]
        test_df = df[df[subject].isin(test_subjects)]

        # Get label ratio of test set
        test_df_label_ratio = test_df[label].value_counts(normalize=True)

        # Calculate difference in label ratio between test set and original df
        ratio_diff = np.abs(test_df_label_ratio - original_df_label_ratio).sum()

        # Check if this is the best split so far
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_train_df = train_df
            best_test_df = test_df
            best_split_label_ratio = test_df_label_ratio

    return best_train_df, best_test_df, original_df_label_ratio, best_split_label_ratio


def create_single_kl_configs(entities_folder, mvs, original_patterns_paths):
    config_template = {
        "type": "single",
        "dataSource": {
            "entitiesPath": None,
            "originalPatternsPaths": None
        },
        "karmaLegoParams": {
            "mvs": None,
            "epsilon": {
                "value": 1,
                "timeUnit": "Milliseconds"
            },
            "maxGap": {
                "value": 250,
                "timeUnit": "Milliseconds"
            },
            "relationSet": "Seven",
            "sacType": "Csac",
            "patternType":  "Complete",
            "maxLevel": 5
        },
        "results": {
            "savePatternsAsIntervals": False,
            "targetPath": None
        }
    }

    configs = []

    # Create testing_configs folder if it doesn't exist
    testing_configs_path = os.path.join(original_patterns_paths, "testing_configs")
    os.makedirs(testing_configs_path, exist_ok=True)

    # Create target_path folder if it doesn't exist
    target_path = os.path.join(original_patterns_paths, "single_kl_results")
    os.makedirs(target_path, exist_ok=True)

    # Find all JSON files in folder
    json_files = [f for f in os.listdir(entities_folder) if f.endswith('.json')]

    for i, json_file in enumerate(json_files):
        # Create a new config based on the template
        config = config_template.copy()

        # Set the entities path to the current JSON file
        entities_path = os.path.join(entities_folder, json_file)
        config['dataSource']['entitiesPath'] = entities_path

        # Set the original patterns path to the input path
        config['dataSource']['originalPatternsPaths'] = [os.path.join(original_patterns_paths, 'patterns.json')]

        # Set the mvs parameter
        config['karmaLegoParams']['mvs'] = mvs

        config['results']['targetPath'] = target_path

        # Add the config to the list
        configs.append(config)

        # Write the config to file
        output_file_path = os.path.join(testing_configs_path, f"single_KL_config_{i+1}.json")
        with open(output_file_path, 'w') as f:
            json.dump(config, f, indent=4)


def merge_csv_files(file1_path, file2_path, output_path):
    # Read the csv files
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # Merge the two dataframes based on the 'Id' column
    merged_df = pd.merge(df1, df2, on='Id')

    # Sort the merged dataframe by the 'Id' column
    merged_df.sort_values('Id', inplace=True)

    # Write the merged dataframe to a new csv file
    merged_df.to_csv(output_path, index=False)


def merge_x_and_y(X_file_path, y_file_path, output_folder, train_or_test, hs_md_hsmd,label):
    # Read X file
    X = pd.read_csv(X_file_path)

    # Extract subject id from X file
    X['subject'] = X['Id'].str[:5]

    # Read labels file
    labels = pd.read_csv(y_file_path)

    # Extract subject id from labels file
    labels['subject'] = labels['Subject'].str[:5]

    # Merge X and labels tables on subject id
    merged = pd.merge(X, labels[['subject', 'Genderio', 'Age', label]], on='subject')

    # Remove rows where label is null
    merged = merged[merged[label].notnull()]


    # Keep only relevant columns
    merged = merged[['Id', 'subject', 'Genderio', 'Age', label] + list(X.columns[1:-1])]

    # Save merged table to output file
    merged.to_csv(f'{output_folder}//X_{train_or_test}_{hs_md_hsmd}_on_{label}.csv', index=False)
