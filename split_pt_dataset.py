import os
import glob
import random

# 数据集根目录（包含 p1_m_pt, p2_m_pt, ...）
root_dir = r"/root/autodl-tmp/pt_dataset"
random.seed(42)

# 找到所有柿子目录
all_obj_dirs = sorted(
    d for d in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, d))
)

num_obj = len(all_obj_dirs)
print("Total objects:", num_obj)

# 打乱柿子顺序（按柿子为单位划分更合理）
random.shuffle(all_obj_dirs)

# 按比例切分柿子目录
n_train = int(num_obj * 0.8)
n_val   = num_obj - n_train

train_objs = all_obj_dirs[:n_train]
val_objs   = all_obj_dirs[n_train:]

print("train/val =", len(train_objs), len(val_objs))

# 收集每个柿子目录下所有 .pt 文件
def collect_files(obj_list):
    paths = []
    for obj in obj_list:
        pt_dir = os.path.join(root_dir, obj)
        files = sorted(glob.glob(os.path.join(pt_dir, "*.pt")))
        paths.extend(files)
    return paths

train_files = collect_files(train_objs)
val_files   = collect_files(val_objs)

print(f"#train files = {len(train_files)}")
print(f"#val   files = {len(val_files)}")

# 写入 txt 文件
def write_list(path, paths):
    with open(path, "w") as f:
        for p in paths:
            f.write(p + "\n")

write_list("pt_train_files.txt", train_files)
write_list("pt_val_files.txt",   val_files)
