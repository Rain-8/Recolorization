import json

# Load the JSON data from the file
with open('Arg.json', 'r') as file:
    data = json.load(file)

# Initialize an empty dictionary to hold the output
output = {}

# Iterate through the entries in the data
for i in range(0, len(data), 20):  # Each original image is followed by 19 target images
    original_entry = data[i]
    original_name = original_entry['name']
    original_palette = original_entry['palette']
    
    # Iterate through the next 19 entries for target images
    for j in range(1, 20):
        target_entry = data[i + j]
        target_name = target_entry['name']
        target_palette = target_entry['palette']
        
        # Create the output structure
        output[target_name] = {
            'src_image_path': original_name,
            'src_palette': original_palette,
            'tgt_image_path': target_name,
            'tgt_palette': target_palette
        }

# Save the output to a new JSON file
with open('output.json', 'w') as outfile:
    json.dump(output, outfile, indent=4)

print("Output has been successfully written to output.json")