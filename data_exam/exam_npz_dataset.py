import os
import shutil
from pathlib import Path

# ==== 配置这里 =====================================================
ROOT_DIR = r"/root/autodl-tmp/npz_dataset_cropped256"
EXPECTED_FOLDER_COUNT = 103
EXPECTED_NPZ_PER_FOLDER = 10201
# ==================================================================


def main():
    root = Path(ROOT_DIR)
    if not root.exists():
        print(f"[ERROR] 根目录不存在: {root}")
        return

    # 只统计第一层子文件夹
    subdirs = sorted([p for p in root.iterdir() if p.is_dir()])

    print("====== 基本统计 ======")
    print(f"1) 子文件夹总数: {len(subdirs)} (期望: {EXPECTED_FOLDER_COUNT})")
    print()

    ok_dirs = []        # 符合“只有 10201 个 npz 且无其他内容”的文件夹
    bad_dirs = []       # 不符合条件的文件夹 (路径, npz数量, 总文件数)
    extra_paths = []    # 所有“多余的文件/目录”（含根目录和子目录中多出来的）

    # 根目录中多余的文件（不是文件夹的东西）
    root_extra = [p for p in root.iterdir() if not p.is_dir()]
    extra_paths.extend(root_extra)

    # 遍历每个子文件夹
    for d in subdirs:
        entries = list(d.iterdir())
        npz_files = [p for p in entries if p.is_file() and p.suffix.lower() == ".npz"]

        total_entries = len(entries)
        npz_count = len(npz_files)

        # 统计多余内容：非 npz 文件 或 子目录
        for p in entries:
            if p.is_dir() or (p.is_file() and p.suffix.lower() != ".npz"):
                extra_paths.append(p)

        # 判断该文件夹是否“刚好 10201 个 npz 且没有其他内容”
        if npz_count == EXPECTED_NPZ_PER_FOLDER and total_entries == npz_count:
            ok_dirs.append(d)
        else:
            bad_dirs.append((d, npz_count, total_entries))

    print("====== 文件夹 npz 数量统计 ======")
    print(
        f"2) 含 {EXPECTED_NPZ_PER_FOLDER} 个 npz 文件且无其他内容的文件夹个数: "
        f"{len(ok_dirs)}"
    )
    print(
        f"3) 不满足条件的文件夹个数: {len(bad_dirs)}"
    )

    if bad_dirs:
        print("   这些文件夹的情况如下：")
        for d, npz_count, total_entries in bad_dirs:
            print(
                f"   - {d.name}: npz 数量 = {npz_count}, "
                f"目录项总数 = {total_entries}"
            )
        print()
        # === 新增：对不满足条件的文件夹进行删除确认 ===
        bad_paths = [d for d, _, _ in bad_dirs]
        print("====== 不满足条件文件夹删除选项 ======")
        print("   d  - 删除所有不满足条件的文件夹（递归删除）")
        print("   s  - 逐个询问是否删除（递归删除）")
        print("   n  - 不删除这些文件夹")
        choice_bad = input("请选择对不满足条件文件夹的操作 (d/s/n): ").strip().lower()

        if choice_bad == "d":
            delete_all(bad_paths)
        elif choice_bad == "s":
            delete_selectively(bad_paths)
        else:
            print("不删除不满足条件的文件夹。")
    else:
        print("   所有文件夹都满足 npz 数量要求。")
        print()

    # 处理多余文件/目录（去重）
    extra_paths = sorted(set(extra_paths), key=lambda p: str(p))

    print("====== 多余文件/目录检查 ======")
    if not extra_paths:
        print("4) 未发现多余的文件或目录，一切干净整齐。")
    else:
        print("4) 检测到以下疑似多余的文件/目录：")
        for idx, p in enumerate(extra_paths, start=1):
            kind = "目录" if p.is_dir() else "文件"
            print(f"   [{idx:03d}] ({kind}) {p}")

        print()
        print("删除选项：")
        print("   y  - 直接删除上面列出的所有内容（包括非空目录）")
        print("   s  - 逐个询问是否删除（包括非空目录）")
        print("   n  - 不删除（只做检查）")

        choice = input("请选择操作 (y/s/n): ").strip().lower()

        if choice == "y":
            delete_all(extra_paths)
        elif choice == "s":
            delete_selectively(extra_paths)
        else:
            print("不删除任何多余文件或目录，仅完成检查。")


# ================================
#   递归删除版本（支持非空目录）
# ================================

def delete_all(paths):
    print("开始递归删除指定文件/目录...")
    for p in paths:
        try:
            if p.is_dir():
                shutil.rmtree(p)
                print(f"   [OK] 已递归删除目录: {p}")
            else:
                p.unlink()
                print(f"   [OK] 已删除文件: {p}")
        except Exception as e:
            print(f"   [ERR] 删除失败: {p}  -> {e}")
    print("删除操作完成。")


def delete_selectively(paths):
    print("逐个确认删除（非空目录也会删除）:")
    for p in paths:
        kind = "目录" if p.is_dir() else "文件"
        ans = input(f"删除 ({kind}) {p}? (y/n): ").strip().lower()
        if ans != "y":
            print(f"   [SKIP] 保留: {p}")
            continue
        try:
            if p.is_dir():
                shutil.rmtree(p)
                print(f"   [OK] 已递归删除目录: {p}")
            else:
                p.unlink()
                print(f"   [OK] 已删除文件: {p}")
        except Exception as e:
            print(f"   [ERR] 删除失败: {p}  -> {e}")
    print("逐个删除流程结束。")


if __name__ == "__main__":
    main()
