#!/usr/bin/env python
"""
Automated Downloader and Extractor for PDBbind Dataset from Zenodo (Record 7724074).
Target Directory: E:\\ResQ-Dock-Data\\
"""

import os
import sys
import time
import tarfile
import urllib.request
from pathlib import Path

ZENODO_URL = "https://zenodo.org/records/7724074/files/PDBBind.tar.gz"
TARGET_DIR = r"E:\ResQ-Dock-Data"
ARCHIVE_PATH = os.path.join(TARGET_DIR, "PDBBind.tar.gz")
EXTRACT_DIR = os.path.join(TARGET_DIR, "PDBBind_raw")


def format_bytes(size: float) -> str:
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def download_with_progress(url: str, output_path: str):
    print(f"Connecting to Zenodo: {url}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    response = urllib.request.urlopen(req)
    total_size = int(response.headers.get('content-length', 0))
    print(f"Total archive size: {format_bytes(total_size)}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    downloaded = 0
    chunk_size = 1024 * 1024 # 1 MB
    t0 = time.time()
    last_print = t0

    # Resume if partially downloaded
    mode = 'wb'
    if os.path.exists(output_path):
        existing_size = os.path.getsize(output_path)
        if existing_size == total_size:
            print("Archive already fully downloaded!")
            return
        elif existing_size < total_size:
            print(f"Resuming download from {format_bytes(existing_size)}...")
            req.headers['Range'] = f"bytes={existing_size}-"
            response = urllib.request.urlopen(req)
            downloaded = existing_size
            mode = 'ab'

    with open(output_path, mode) as f:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            if now - last_print > 3.0:
                speed = downloaded / max(now - t0, 1e-5)
                pct = (downloaded / total_size) * 100.0 if total_size > 0 else 0.0
                print(f"Downloaded: {format_bytes(downloaded)} / {format_bytes(total_size)} ({pct:.1f}%) | Speed: {format_bytes(speed)}/s")
                last_print = now

    print(f"\nDownload completed successfully in {time.time() - t0:.1f}s -> {output_path}")


def extract_archive(archive_path: str, extract_dir: str):
    print(f"\nExtracting {archive_path} to {extract_dir}...")
    os.makedirs(extract_dir, exist_ok=True)
    t0 = time.time()
    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()
        print(f"Archive contains {len(members)} entries. Extracting...")
        tar.extractall(path=extract_dir)
    print(f"Extraction completed in {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    download_with_progress(ZENODO_URL, ARCHIVE_PATH)
    extract_archive(ARCHIVE_PATH, EXTRACT_DIR)

    # Inspect contents
    found_dirs = [d for d in os.listdir(EXTRACT_DIR) if os.path.isdir(os.path.join(EXTRACT_DIR, d))]
    print(f"\nTotal items extracted: {len(found_dirs)} folders in {EXTRACT_DIR}")
    if found_dirs:
        print("Sample folders:", found_dirs[:10])
