"""Download the hackathon dataset from the organiser's Google Drive into data/."""
import urllib.request
from pathlib import Path

FILES = {
    "corpus.jsonl": "1g-XApuwcZfD2tYF6di9QLmFFl_TPe5OO",
    "eval_public.jsonl": "1wpIdvUDOJDfG5c1A_o6IXi_uHpRTY8cG",
    "eval_hidden.jsonl": "1L-3Aum_nZ0FsGwVpWv_IpzREzALYRm2E",
    "dataset-README.md": "1_HCR_uvmqsE4o-z72bUnzKfXUdo9IxyO",
}

if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    for name, file_id in FILES.items():
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        print("downloading", name)
        urllib.request.urlretrieve(url, Path("data") / name)
    print("done")
