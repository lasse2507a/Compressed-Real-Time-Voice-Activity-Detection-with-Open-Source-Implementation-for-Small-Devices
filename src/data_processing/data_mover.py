import os
import random
import shutil

def move_random_npy_files(source_dir, dest_dir, num_files):
    """
    Move a specified number of randomly selected .npy files from a source directory to a destination directory.
    Args:
        source_dir (str): The directory containing the source .npy files.
        dest_dir (str): The directory to move the selected .npy files to.
        num_files (int): The number of randomly selected .npy files to move.
    """
    npy_files = [f for f in os.listdir(source_dir) if f.endswith('.npy')]
    random_files = random.sample(npy_files, num_files)
    for file in random_files:
        src_path = os.path.join(source_dir, file)
        dest_path = os.path.join(dest_dir, file)
        shutil.move(src_path, dest_path)
    print(f"moved {num_files} .npy files from {source_dir} to {dest_dir}.")


if __name__ == "__main__":
    move_random_npy_files(source_dir='data/output/training_clip_len_17200samples/mfsc_window_400samples',
                          dest_dir='data/output/validation_clip_len_17200samples/mfsc_window_400samples',
                          num_files=25000)
