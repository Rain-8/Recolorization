import json
import random

def split_json_data(file_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Load JSON data
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Get all keys from the JSON data
    keys = list(data.keys())
    
    # Shuffle keys
    random.shuffle(keys)
    
    # Compute split indices
    total_keys = len(keys)
    train_end = int(total_keys * train_ratio)
    val_end = train_end + int(total_keys * val_ratio)
    
    # Split the keys
    train_keys = keys[:train_end]
    val_keys = keys[train_end:val_end]
    test_keys = keys[val_end:]
    
    # Create dictionaries for each split
    train_data = {key: data[key] for key in train_keys}
    val_data = {key: data[key] for key in val_keys}
    test_data = {key: data[key] for key in test_keys}
    
    # Return the split data
    return train_data, val_data, test_data

# Example usage in main
if __name__=="__main__":
    train_data, val_data, test_data = split_json_data('../../datasets/processed_palettenet_data_sample_v2/output.json')
    output_dir = "../../datasets/processed_palettenet_data_sample_v2"
    # Save each split to a new JSON file
    with open(f'{output_dir}/train.json', 'w') as f:
        json.dump(train_data, f, indent=4)

    with open(f'{output_dir}/val.json', 'w') as f:
        json.dump(val_data, f, indent=4)

    with open(f'{output_dir}/test.json', 'w') as f:
        json.dump(test_data, f, indent=4)

    print("Data has been split into train, validation, and test sets.")
