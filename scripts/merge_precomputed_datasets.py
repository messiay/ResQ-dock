#!/usr/bin/env python
"""
Merges processed_436 and processed_scale into a single unified directory
E:\\ResQ-Dock-Data\\processed_merged\\ for maximum training scale.
"""

import os
import glob
import shutil

DIR_A = r"E:\ResQ-Dock-Data\processed_436"
DIR_B = r"E:\ResQ-Dock-Data\processed_scale"
MERGED_DIR = r"E:\ResQ-Dock-Data\processed_merged"

os.makedirs(MERGED_DIR, exist_ok=True)

files_a = glob.glob(os.path.join(DIR_A, "*.pt"))
files_b = glob.glob(os.path.join(DIR_B, "*.pt"))

copied = 0
for f in files_a + files_b:
    dest = os.path.join(MERGED_DIR, os.path.basename(f))
    if not os.path.exists(dest):
        shutil.copyfile(f, dest)
        copied += 1

total = len(glob.glob(os.path.join(MERGED_DIR, "*.pt")))
print(f"Dataset merge complete!")
print(f"Added {copied} new files.")
print(f"Total unified crystal complexes ready in {MERGED_DIR}: {total}")
