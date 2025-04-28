from huggingface_hub import snapshot_download
import time

# because of rate limiting, you have to rerun this script from time to time
while True:
    try:
        ret = snapshot_download(
            repo_id="dasgringuen/assettoCorsaGym",
            repo_type="dataset",
            local_dir="AssettoCorsaGymDataSet",
            allow_patterns="data_sets/ks_red_bull_ring-layout_gp/bmw_z4_gt3/*",
            resume_download=True,
        )
        if "AssettoCorsaGymDataSet" in ret:
            print("Rate limit exceeded. Waiting for 1 minute...")
            time.sleep(60)
            continue
    except Exception as e:
        print(f"Retrying in 1 minute...")
        time.sleep(60)
        continue