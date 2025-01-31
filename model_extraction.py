import os
import csv
from tqdm import tqdm
import shutil

# Directory containing decomposed files
decomposed_dir = './decomposed_apk/'
model_dir = './extracted_models/'
csv_file = 'dl_apks.csv'

# Ensure the model directory exists
os.makedirs(model_dir, exist_ok=True)

# Define the file suffixes for various deep learning frameworks
dl_suffixes = {
    "TensorFlow Lite": [".tflite", ".lite", ".tfl"]
}


def find_smali_references(decomposed_dir, model_name):
    """Search for any .smali files containing references to the model file within directories starting with 'smali'."""
    for root, dirs, files in os.walk(decomposed_dir):
        # Continue only if the current directory starts with 'smali'
        if 'smali' in root.split('/'):
            for file in files:
                if file.endswith(".smali"):
                    with open(os.path.join(root, file), "r") as smali_file:
                        if model_name in smali_file.read():
                            return os.path.join(root, file)
    return None


def handle_duplicate_file_names(path):
    """Check if a file exists and modify the filename to avoid overwriting by appending a number."""
    original_path = path
    counter = 1
    while os.path.exists(path):
        path = f"{original_path.rsplit('.', 1)[0]}_{counter}.{original_path.rsplit('.', 1)[1]}"
        counter += 1
    return path


def find_dl_apks(decomposed_dir):
    """Scan directories, identify DL APKs, and write results to a CSV file."""

    dl_count = 0

    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['APK_sha256', 'Model', 'Framework', 'Smali_path'])

        subdirs = [os.path.join(decomposed_dir, subdir) for subdir in os.listdir(decomposed_dir) if
                   os.path.isdir(os.path.join(decomposed_dir, subdir))]

        for subdir_path in tqdm(subdirs, desc="Scanning subdirectories"):
            dl_apk = False

            for root, _, files in os.walk(subdir_path):
                for file in files:
                    for framework, suffixes in dl_suffixes.items():
                        if any(file.endswith(suffix) for suffix in suffixes):

                            smali_reference = find_smali_references(subdir_path, file)
                            writer.writerow([subdir_path.split("/")[-1], file, framework, smali_reference])
                            print(f"Detected {file} in {framework} at {subdir_path}")
                            print(f"Found {file} in {smali_reference}")
                            model_filename = subdir_path.split("/")[-1] + '_' + file
                            full_model_path = os.path.join(model_dir, model_filename)
                            full_model_path = handle_duplicate_file_names(full_model_path)
                            shutil.copy2(os.path.join(root, file), full_model_path)

                            if not dl_apk:  # Count this subdir as containing a DL APK only once
                                dl_count += 1
                                print(f'Found DL APKs: {dl_count}')
                                dl_apk = True
                                break  # Stop after recording this file to avoid multiple entries for the same framework.

            if not dl_apk:  # If no model found, delete the subdir
                shutil.rmtree(subdir_path)
                print(f"Deleted non-DL APK directory {subdir_path}")

    return dl_count


if __name__ == "__main__":
    total_dl_apks = find_dl_apks(decomposed_dir)
    print(
        f'Scanning complete. Total number of DL APKs: {total_dl_apks}. Results are stored in {csv_file}.')
