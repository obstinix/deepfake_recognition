import os
import cv2
import argparse
import random
import shutil
from tqdm import tqdm

def main(raw_dir, processed_dir, split_ratios):
    # Setup directories
    for split in ["train", "val", "test"]:
        for class_name in ["real", "fake"]:
            os.makedirs(os.path.join(processed_dir, split, class_name), exist_ok=True)

    for class_name in ["real", "fake"]:
        src_dir = os.path.join(raw_dir, class_name)
        if not os.path.exists(src_dir):
            print(f"Directory not found: {src_dir}")
            continue

        files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.mp4'))]
        random.shuffle(files)

        n_files = len(files)
        train_end = int(n_files * split_ratios[0])
        val_end = train_end + int(n_files * split_ratios[1])

        splits = {
            "train": files[:train_end],
            "val": files[train_end:val_end],
            "test": files[val_end:]
        }

        for split, split_files in splits.items():
            for f in tqdm(split_files, desc=f"Processing {class_name} ({split})"):
                src_path = os.path.join(src_dir, f)
                
                # If video, extract frames (simple implementation - extracts first frame)
                if f.lower().endswith('.mp4'):
                    cap = cv2.VideoCapture(src_path)
                    ret, frame = cap.read()
                    if ret:
                        dst_path = os.path.join(processed_dir, split, class_name, f"{os.path.splitext(f)[0]}.jpg")
                        cv2.imwrite(dst_path, frame)
                    cap.release()
                else:
                    dst_path = os.path.join(processed_dir, split, class_name, f)
                    shutil.copy2(src_path, dst_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', required=True, help="Directory with raw data (must contain 'real' and 'fake' subdirs)")
    parser.add_argument('--processed_dir', default='data', help="Output directory for processed data")
    parser.add_argument('--splits', nargs=3, type=float, default=[0.7, 0.15, 0.15], help="Train/Val/Test split ratios")
    args = parser.parse_args()
    
    assert sum(args.splits) == 1.0, "Split ratios must sum to 1.0"
    main(args.raw_dir, args.processed_dir, args.splits)
