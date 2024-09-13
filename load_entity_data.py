import os
import numpy as np

def load_entity_data(folder_path):
    entity_data = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".eea"):
            file_path = os.path.join(folder_path, file_name)

            # Read data from .eea file
            with open(file_path, 'r') as file:
                content = file.read()
                # Split the content by whitespace (spaces and newlines)
                numeric_values = list(map(float, content.split()))
            
            # Divide the data into channels, each channel containing 7680 samples
            num_samples_per_channel = 7680
            num_channels = len(numeric_values) // num_samples_per_channel

            # Check if the number of channels is exactly 16
            if num_channels != 16:
                raise ValueError(f"File {file_name} does not have exactly 16 channels. Found: {num_channels} channels.")
            
            data = np.array(numeric_values).reshape(num_channels, num_samples_per_channel)
            
            # Use file name (without extension) as entity name and subject
            entity_name = os.path.splitext(file_name)[0]
            subject = entity_name
            session = 1

            entity_data.append({
                "entity_name": entity_name,
                "subject": subject,
                "session": session,
                "data": data
            })

    # Generate channel names based on number of channels
    channel_names = [f"Channel_{i+1}" for i in range(num_channels)]

    return entity_data, channel_names


