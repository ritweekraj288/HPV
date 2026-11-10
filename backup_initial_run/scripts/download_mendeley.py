import os
import urllib.request
import json

FILES = [
    ("Dataset 1_Epidemiology Stats.xlsx", "994d932f-a3da-4e85-9cae-59ddfd8947e8"),
    ("Dataset 2_Association analysis.xlsx", "97858c77-bf6e-47de-8adc-c81e280037d5"),
    ("Dataset 3_Cor_ANOVA_Reg_interaction effects.xlsx", "49aae2fe-857d-4eda-b32e-3b7f64d6e653"),
    ("Dataset 4_PCA_Kmeans_MachineLearning.xlsx", "b82d5a10-7ae3-4dee-b4d1-5890e6bc7f99"),
    ("Table S1.xlsx", "13f46318-01e6-4e61-aff9-0c9a5c8b6eb8")
]

dest_dir = "/Users/ritweekraj/Desktop/HPV/data/mendeley"
os.makedirs(dest_dir, exist_ok=True)

for fname, file_id in FILES:
    dest_path = os.path.join(dest_dir, fname)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        print(f"Already exists: {fname} ({os.path.getsize(dest_path)} bytes)")
        continue
    
    print(f"Fetching metadata for {fname}...")
    api_url = f"https://data.mendeley.com/public-api/datasets/pn7h6yjt6k/files/{file_id}"
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with urllib.request.urlopen(req) as resp:
        meta = json.loads(resp.read().decode())
    
    download_url = meta.get("content_details", {}).get("download_url")
    if not download_url:
        download_url = f"https://data.mendeley.com/public-files/datasets/pn7h6yjt6k/files/{file_id}/file_downloaded"
    
    print(f"Downloading {fname} from {download_url}...")
    dl_req = urllib.request.Request(download_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(dl_req) as resp, open(dest_path, "wb") as out_file:
        out_file.write(resp.read())
    print(f"Saved {fname} ({os.path.getsize(dest_path)} bytes)")

print("All Mendeley files downloaded successfully!")
