import os
import numpy as np
import zipfile

ROOT = "/root/autodl-tmp/npz dataset"  # 改成你的 npz 根目录

bad_files = []

for root, _, files in os.walk(ROOT):
    for f in files:
        if f.endswith(".npz"):
            path = os.path.join(root, f)
            try:
                np.load(path, allow_pickle=False)
            except zipfile.BadZipFile:
                print("[BAD ZIP]", path)
                bad_files.append(path)
            except Exception as e:
                print("[OTHER ERROR]", path, e)
                bad_files.append(path)

print("\n===== SUMMARY =====")
print("Total bad files:", len(bad_files))
