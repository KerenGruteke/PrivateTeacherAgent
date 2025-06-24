import zipfile
import os
import json
import csv

def unpack_and_preview_history_data(zip_path, extract_to='history_data', csv_path='history_examples.csv', num_examples=3):
    # Unpack the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted files to {extract_to}")

    examples = []
    # Find all json files in the extracted directory
    for root, dirs, files in os.walk(extract_to):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        print(f"Example from {file_path}:")
                        print(json.dumps(data, indent=2)[:500])  # Print first 500 chars
                        print('-' * 40)
                        examples.append(data)
                        if len(examples) >= num_examples:
                            break
                except Exception as e:
                    print(f"Could not read {file_path}: {e}")
        if len(examples) >= num_examples:
            break

    # Save a few data examples to a CSV file
    if examples:
        # Flatten the JSON if possible, otherwise just save as string
        keys = set()
        for ex in examples:
            if isinstance(ex, dict):
                keys.update(ex.keys())
        keys = list(keys) if keys else ['data']

        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            for ex in examples:
                if isinstance(ex, dict):
                    writer.writerow({k: ex.get(k, "") for k in keys})
                else:
                    writer.writerow({'data': json.dumps(ex)})
        print(f"Saved {len(examples)} examples to {csv_path}")

# Example usage:
# unpack_and_preview_history_data('history_data.zip')

