import os
import glob
import random

# 数据集根目录
root_dir = r"/root/autodl-tmp/npz_dataset_cropped256"
random.seed(42)


all_obj_dirs = sorted(
    d for d in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, d))
)

num_obj = len(all_obj_dirs)
print("Total objects:", num_obj)

# 打乱顺序
random.shuffle(all_obj_dirs)

# 按比例切分
n_train = int(num_obj * 0.8)
n_val   = num_obj - n_train
# n_test  = num_obj - n_train - n_val

train_objs = all_obj_dirs[:n_train]
val_objs   = all_obj_dirs[n_train:n_train+n_val]

print("train/val =", len(train_objs), len(val_objs))

def collect_files(obj_list):
    paths = []
    for obj in obj_list:
        npz_dir = os.path.join(root_dir, obj)
        files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
        paths.extend(files)
    return paths

train_files = collect_files(train_objs)
val_files   = collect_files(val_objs)

print(f"#train files = {len(train_files)}")
print(f"#val   files = {len(val_files)}")


def write_list(path, paths):
    with open(path, "w") as f:
        for p in paths:
            f.write(p + "\n")


write_list("train_files.txt", train_files)
write_list("val_files.txt",   val_files)
