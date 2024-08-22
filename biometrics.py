import pandas as pd
import numpy as np
import os

def process_data_files(folder_path, file_names, output_file):
    # Initialize dictionaries to store data and sampling frequencies
    data_dict = {}
    sample_freq_dict = {}

    # Load data from each file and extract sampling frequency
    for name in file_names:
        file_path = os.path.join(folder_path, f"{name}.csv")
        df = pd.read_csv(file_path, header=None)

        # Extract sampling frequency from the second row
        sample_freq = int(df.iloc[1, 0])
        sample_freq_dict[name] = sample_freq

        # Extract data starting from the third row
        data = df.iloc[2:].reset_index(drop=True)
        data_dict[name] = data[0].values

    # Initialize lists to store processed data and track indices
    processed_data = []
    time = 0
    indices = {name: 0 for name in file_names}

    # Process data based on sampling frequencies
    while any(indices[name] < len(data_dict[name]) for name in file_names):
        entry = [time]
        for name in file_names:
            sample_freq = sample_freq_dict[name]
            values = []
            for _ in range(sample_freq):
                if indices[name] < len(data_dict[name]):
                    values.append(data_dict[name][indices[name]])
                    indices[name] += 1
                else:
                    break
            avg_value = np.mean(values) if values else np.nan
            entry.append(avg_value)

        processed_data.append(entry)
        time += 1

    # Create DataFrame and save it to CSV
    columns = ['Time'] + file_names
    output_df = pd.DataFrame(processed_data, columns=columns)
    output_df.to_csv(output_file, index=False)

# Example usage
folder_path = r'C:\Users\marko\PycharmProjects\cogload\biometdata\Data\S1\Final'
file_names = ['HR', 'EDA', 'TEMP']
output_file = 'output.csv'
process_data_files(folder_path, file_names, output_file)
