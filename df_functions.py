import os
import pandas as pd
import os
import random
import shutil


def split_data(input_folder, train_folder, test_folder, split_ratio=0.75):
    file_list = os.listdir(input_folder)
    num_files = len(file_list)
    num_train = int(num_files * split_ratio)
    train_files = set(random.sample(file_list, num_train))

    for file_name in file_list:
        if file_name in train_files:
            shutil.copy(os.path.join(input_folder, file_name), os.path.join(train_folder, file_name))
        else:
            shutil.copy(os.path.join(input_folder, file_name), os.path.join(test_folder, file_name))

def merge_excel_files(main_folder):
    # Find all subfolders in the main folder
    subfolders = [subfolder for subfolder in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, subfolder))]

    # Initialize empty dataframes to store the merged data
    csv_files_path = os.path.join(main_folder, subfolders[0], 'experiment_1')
    hs_file = os.path.join(csv_files_path, 'horizontalSupport.csv')
    md_file = os.path.join(csv_files_path, 'meanDuration.csv')
    hs_df = pd.read_csv(hs_file, index_col=0)
    md_df = pd.read_csv(md_file, index_col=0)


    # Loop through the subfolders and merge the data from the excel files
    for subfolder in subfolders[1:]:
        csv_files_path = os.path.join(main_folder, subfolder, 'experiment_1')
        hs_file = os.path.join(csv_files_path, 'horizontalSupport.csv')
        md_file = os.path.join(csv_files_path, 'meanDuration.csv')

        if os.path.isfile(hs_file) and os.path.isfile(md_file):
            # Load the data from the excel files
            hs_data = pd.read_csv(hs_file, index_col=0)
            md_data = pd.read_csv(md_file, index_col=0)

            # Check if columns in new_df and combined_df match
            if set(hs_data.columns) == set(hs_df.columns):
                hs_data = hs_data[hs_df.columns]  # reorder the columns
                # Append data to the main dataframes
                hs_df = pd.concat([hs_df, hs_data],axis=0, ignore_index=False)
            else:
                print(f"Columns in {hs_file} do not match, skipping...")

            if set(md_data.columns) == set(md_df.columns):
                md_data = md_data[md_df.columns]  # reorder the columns
                # Append data to the main dataframes
                md_df = pd.concat([md_df, md_data], axis=0, ignore_index=False)
            else:
                print(f"Columns in {md_file} do not match, skipping...")
        else:
            print(f"Could not find {hs_file} or {md_file}, skipping...")

        # Save merged dataframes to excel files
    hs_output = os.path.join(main_folder, "all_singles_horizontalSupport.csv")
    md_output = os.path.join(main_folder, "all_singles_meanDuration.csv")
    hs_df.to_csv(hs_output, index=True)
    md_df.to_csv(md_output, index=True)

    print("Merged data saved successfully!")


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


def merge_train_test(train_file, test_file):
    # Read in train and test CSV files
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Check if train and test have the same columns
    if set(train_df.columns) != set(test_df.columns):
        print("Error: train and test have different columns")
        return None

    # Merge train and test into a single DataFrame
    merged_df = pd.concat([train_df, test_df], axis=0)

    return merged_df


def merge_X_and_y_HDRS21(x_file_path,y_file_path,output_file_path):

    # Load X and y CSV files into dataframes
    X = pd.read_csv(x_file_path)
    y = pd.read_csv(y_file_path)

    # Extract subject code from Id column in X dataframe
    X['subject'] = X['Id'].apply(lambda x: x[:5])
    X['sess'] = X['Id'].apply(lambda x: x[10])
    y['end_score']=y['BL_HDRS21_totalscore']+y['HDRS21_Change']
    # Create new Age and Genderio columns in X dataframe
    X['Age'] = ''
    X['Genderio'] = ''

    # Fill Age and Genderio columns in X dataframe from y dataframe
    X.loc[X['sess'] == '1', 'Age'] = X.loc[X['sess'] == '1', 'subject'].map(y.set_index('Subject')['Age'])
    X.loc[X['sess'] == '1', 'Genderio'] = X.loc[X['sess'] == '1', 'subject'].map(
        y.set_index('Subject')['Genderio'])

    # Fill Age and Genderio columns in X dataframe from y dataframe
    X.loc[X['sess'] == '2', 'Age'] = X.loc[X['sess'] == '2', 'subject'].map(y.set_index('Subject')['Age'])
    X.loc[X['sess'] == '2', 'Genderio'] = X.loc[X['sess'] == '2', 'subject'].map(
        y.set_index('Subject')['Genderio'])

    # Create separate dataframes for sess1 and sess2 records
    sess1_df = X[X['sess'] == '1']
    sess2_df = X[X['sess'] == '2']

    # Add BL_HDRS21_totalscore column to sess1 dataframe
    sess1_df['label'] = sess1_df['subject'].map(y.set_index('Subject')['BL_HDRS21_totalscore'])

    # Add SC_HDRS21_totalscore column to sess2 dataframe
    sess2_df['label'] = sess2_df['subject'].map(y.set_index('Subject')['end_score'])

    # Concatenate sess1 and sess2 dataframes back into original X dataframe
    X = pd.concat([sess1_df, sess2_df])

    # Drop the subject column as it's no longer needed
    X = X.drop(columns=['subject', 'sess'], axis=1)

    X.to_csv(output_file_path, index=False)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def train_test_split_by_subject(df,label,subject,test_size=0.3, random_state=42):
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

def filter_by_percentile(df, column):
    column_data = df[column]
    threshold_high = column_data.quantile(0.75)
    threshold_low = column_data.quantile(0.25)
    filtered_df = df[(column_data > threshold_high) | (column_data < threshold_low)]
    return filtered_df


def create_mean_df(df):
    # Create new columns
    df['subject'] = df['Id'].str[:5]
    df['sess'] = df['Id'].str[10]
    df['part'] = df['Id'].str[12:]

    # Group by subject, sess, and part, and calculate the mean for each group
    mean_df = df.groupby(['subject', 'sess', 'part']).mean().reset_index()

    return mean_df


def preprocess_df(df):
    # Extract subject, sess, and part information from the 'Id' column
    df['subject'] = df['Id'].apply(lambda x: x[:5])
    df['sess'] = df['Id'].apply(lambda x: x[10])
    df['part'] = df['Id'].apply(lambda x: x[12:])

    # Create a new dataframe with the mean of all parts from each unique [subject, sess] combination
    new_df = df.groupby(['subject', 'sess']).mean().reset_index()

    return new_df


def preprocess_df2(df):

    # create a list of subjects that have both sess=1 and sess=2
    valid_subjects = []
    for subject in df['subject'].unique():
        if ('1' in df[df['subject'] == subject]['sess'].values) and ('2' in df[df['subject'] == subject]['sess'].values):
            valid_subjects.append(subject)

    # create a new df with the difference between sessions for each subject
    diff_df = pd.DataFrame()
    for subject in valid_subjects:
        sess1 = df[(df['subject'] == subject) & (df['sess'] == '1')]
        sess2 = df[(df['subject'] == subject) & (df['sess'] == '2')]
        diff = pd.DataFrame(data=(sess2.iloc[:, 2:-4].values - sess1.iloc[:, 2:-4].values),
                            columns=sess2.columns[2:-4].values)
        diff['subject'] = subject  # add the subject column
        diff['Age'] = float(sess2['Age'])
        diff['Genderio'] = float(sess2['Genderio'])
        diff['label'] = float(sess2.iloc[:, -1]) - float(sess1.iloc[:, -1])
        diff_df = pd.concat([diff_df, diff], axis=0)

    return diff_df