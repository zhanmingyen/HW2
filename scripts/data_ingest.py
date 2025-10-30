import os
import requests

DATA_URL = "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def download_dataset(dest_path: str):
    print(f"Downloading dataset from {DATA_URL} to {dest_path}...")
    r = requests.get(DATA_URL, timeout=30)
    r.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(r.content)
    print("Download complete.")


def main():
    ensure_data_dir()
    dest = os.path.join(DATA_DIR, "sms_spam_no_header.csv")
    if os.path.exists(dest):
        print(f"Dataset already exists at {dest}")
    else:
        download_dataset(dest)


if __name__ == "__main__":
    main()
