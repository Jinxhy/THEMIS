import os
import subprocess
import logging
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


def reassemble_apk(source_dir, output_apk_path):
    try:
        subprocess.run(['./apktool', 'b', source_dir, '-o', output_apk_path, '-f'], check=True, capture_output=True, text=True)
        logging.info(f"Successfully reassembled: {source_dir} -> {output_apk_path}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to reassemble: {source_dir}, Error: {str(e)}")
        return False


def reassemble_apks_concurrently(input_dir, output_dir, max_workers=200):
    source_dirs = [os.path.join(input_dir, name) for name in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, name))]
    if not source_dirs:
        logging.info("No directories found in the input directory.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_dir = {
            executor.submit(reassemble_apk, src_dir, os.path.join(output_dir, os.path.basename(src_dir) + '.apk')): src_dir
            for src_dir in source_dirs
        }
        for future in as_completed(future_to_dir):
            src_dir = future_to_dir[future]
            try:
                future.result()
            except Exception as exc:
                logging.error(f"Exception occurred for {src_dir}: {exc}")


if __name__ == "__main__":
    input_directory = './decomposed_apk/'  # Directory with decomposed APK folders
    output_directory = './reassembled_apk/'  # Output directory for reassembled APKs
    max_workers = 50  # Adjust based on system capacity
    log_file = 'reassembled_log.txt'

    setup_logging(log_file)
    reassemble_apks_concurrently(input_directory, output_directory, max_workers)
