import os

ROOT_DIR = r"/root/autodl-tmp/npz dataset"  # 改成你的根目录

def sizeof_fmt(num, suffix="B"):
    """人类可读的文件大小"""
    for unit in ["", "K", "M", "G", "T"]:
        if num < 1024:
            return f"{num:.2f}{unit}{suffix}"
        num /= 1024
    return f"{num:.2f}P{suffix}"


def folder_size(folder):
    """返回文件夹下所有文件的总字节数"""
    total = 0
    for root, _, files in os.walk(folder):
        for f in files:
            path = os.path.join(root, f)
            total += os.path.getsize(path)
    return total


def main():
    if not os.path.isdir(ROOT_DIR):
        print(f"[ERROR] ROOT_DIR not found: {ROOT_DIR}")
        return

    # 找所有 *_npz 的子文件夹
    subdirs = [
        os.path.join(ROOT_DIR, d)
        for d in os.listdir(ROOT_DIR)
        if os.path.isdir(os.path.join(ROOT_DIR, d)) and d.endswith("_npz")
    ]

    if not subdirs:
        print(f"[ERROR] No *_npz directories found under {ROOT_DIR}")
        return

    print(f"Found {len(subdirs)} npz folders:\n")
    grand_total = 0

    for folder in subdirs:
        size_bytes = folder_size(folder)
        grand_total += size_bytes
        print(f"{folder}  →  {sizeof_fmt(size_bytes)}")

    print("\n" + "#" * 50)
    print(f"Total size of all folders: {sizeof_fmt(grand_total)}")
    print("#" * 50)


if __name__ == "__main__":
    main()
