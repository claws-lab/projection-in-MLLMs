import os, sys
import csv
import json
import random
import pickle as pkl

def get_human_message():
    classes = [
        "... ", # fill in with your label space as a list of strings
        "insert your classes here",
        "..."
    ]
    classes = ['"' + x + '"' for x in classes]
    random.shuffle(classes)
    classes_string = ', '.join(classes)
    human_message = "<image>\nClassify this image into one of the following categories relating to plant leaf diseases: " + classes_string + ". Only output a single final classification label and NOTHING ELSE.\n\nOutput:"
    return human_message

# Function to generate the JSON structure
def generate_json(csv_filename):
    data = []
    id_counter = 0
    with open(csv_filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Generate a unique ID
            id = str(id_counter).zfill(9)  # Pad with zeros to maintain a consistent ID length
            # Randomly select a human message
            human_message = get_human_message()
            image_path = row['imagefilename']
            # Generate the JSON structure for each row
            item = {
                "id": id,
                "image": image_path,
                "conversations": [
                    {
                        "from": "human",
                        "value": human_message
                    },
                    {
                        "from": "gpt",
                        "value": f"{row['label']}"
                    }
                ]
            }
            data.append(item)
    return json.dumps(data, indent=2)

def create_labelled_csv(directory, output_csv='output.csv'):
    labelled_images = []

    # Traverse through each subdirectory in the given directory
    for label in os.listdir(directory):
        subdirectory = os.path.join(directory, label)
        
        # Check if it's a directory
        if os.path.isdir(subdirectory):
            modified_label = label.strip().replace('_', '')
            for imagefile in os.listdir(subdirectory):
                # Construct full image file path
                image_path = os.path.join(subdirectory, imagefile)
                # Append the image file path and modified label to the list
                if os.path.isfile(image_path):
                    labelled_images.append((image_path, modified_label))

    # Shuffle the list
    random.shuffle(labelled_images)

    # Write to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['imagefilename', 'label'])
        writer.writerows(labelled_images)



directory = '<insert directory pat here>' # Replace with your directory path containing the labelled images
create_labelled_csv(directory, output_csv='tuning_data_with_labels.csv')
csv_file = 'tuning_data_with_labels.csv'
json_output = generate_json(csv_file)

# Write the JSON output to a file
with open('tuning_data_with_labels.json', 'w') as jsonfile:
    jsonfile.write(json_output)
