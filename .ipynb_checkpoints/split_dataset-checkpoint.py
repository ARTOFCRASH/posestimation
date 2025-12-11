import os
import glob
import random
import shutil

# ------------------- 配置 -------------------
root_dir = r"/root/autodl-tmp/npz_dataset_cropped256"  # 原数据集
out_root = r"/root/autodl-tmp/dataset_split"          # 目标输出

train_dir = os.path.join(out_root, "train")
val_dir   = os.path.join(out_root, "val")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

random.seed(42)
# ---------------------------------------------


# 1) 找到所有对象目录
all_obj_dirs = sorted(
    d for d in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, d))
)
print("Total objects:", len(all_obj_dirs))

# 2) 随机打乱
random.shuffle(all_obj_dirs)

# 3) 8:2 划分
n_train = int(len(all_obj_dirs) * 0.8)

train_objs = all_obj_dirs[:n_train]
val_objs   = all_obj_dirs[n_train:]

print("train/val objects =", len(train_objs), len(val_objs))


# 4) 辅助函数：复制文件并解决重名问题
def copy_files(obj_list, dst_folder):
    count = 0
    for obj in obj_list:
        obj_dir = os.path.join(root_dir, obj)
        npz_files = sorted(glob.glob(os.path.join(obj_dir, "*.npz")))

        for src_path in npz_files:
            fname = os.path.basename(src_path)
            dst_path = os.path.join(dst_folder, fname)

            # 文件重名自动加对象名前缀
            if os.path.exists(dst_path):
                fname = f"{obj}_{fname}"
                dst_path = os.path.join(dst_folder, fname)

            shutil.copy2(src_path, dst_path)
            count += 1

    return count


# 5) 复制 train/val 数据
train_count = copy_files(train_objs, train_dir)
val_count   = copy_files(val_objs,   val_dir)

print(f"Copied {train_count} train files → {train_dir}")
print(f"Copied {val_count} val files   → {val_dir}")
print("\nDone.")
