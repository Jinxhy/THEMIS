import os
import subprocess
import logging
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed


def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def decompile_apk(apk_path, output_dir):
    apk_name = os.path.basename(apk_path).replace('.apk', '')
    output_path = os.path.join(output_dir, apk_name)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    try:
        subprocess.run(['./apktool', 'd', apk_path, '-o', output_path, '-f'], check=True, capture_output=True,
                       text=True)
        logging.info(f"Successfully decomposed: {apk_path}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to decompose: {apk_path}, Error: {str(e)}")
        return False


def decompile_apks_concurrently(input_dir, output_dir, max_workers=200):
    apk_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith('.apk')]
    if not apk_files:
        logging.info("No APK files found in the input directory.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_apk = {executor.submit(decompile_apk, apk, output_dir): apk for apk in apk_files}
        for future in as_completed(future_to_apk):
            apk = future_to_apk[future]
            try:
                future.result()
            except Exception as exc:
                logging.error(f"Exception occurred for {apk}: {exc}")


if __name__ == "__main__":
    input_directory = './download_apk/'
    output_directory = './decomposed_apk/'
    max_workers = 50  # Adjust this number based on your system's capability
    log_file = 'decomposed_log.txt'

    setup_logging(log_file)
    decompile_apks_concurrently(input_directory, output_directory, max_workers)
