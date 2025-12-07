import os
import glob
import re
from pathlib import Path
from collections import defaultdict

# 根目录，下面有 p4_m_npz, p7_m_npz 等子目录
ROOT_DIR = r"/root/autodl-tmp/npz dataset"

# 要检查的子文件夹名字
TARGET_DIRS = ["p22_l_npz"]

# 期望的 roll / pitch 范围
ROLL_MIN, ROLL_MAX = -50, 50
PITCH_MIN, PITCH_MAX = -50, 50

# 匹配类似 "p4_m_-10_15_" 这样的 stem
stem_pattern = re.compile(
    r'^(?P<prefix1>[^_]+)_(?P<prefix2>[^_]+)_(?P<roll>-?\d+)_(?P<pitch>-?\d+)_?$'
)

def parse_fname(fname: str):
    """
    从文件名中解析 prefix1, roll, pitch.
    输入: 纯文件名, 如 "p4_m_-10_15_.npz"
    返回: (prefix1:str, roll:int, pitch:int) 或 None
    """
    stem = os.path.splitext(fname)[0]  # 去掉 .npz
    m = stem_pattern.match(stem)
    if m is None:
        print(f"[WARN] 文件名不符合格式: {fname}")
        return None
    prefix1 = m.group("prefix1")
    roll = int(m.group("roll"))
    pitch = int(m.group("pitch"))
    return prefix1, roll, pitch


def main():
    root = Path(ROOT_DIR)

    # 理论上的完整集合
    expected = set(
        (r, p)
        for r in range(ROLL_MIN, ROLL_MAX + 1)
        for p in range(PITCH_MIN, PITCH_MAX + 1)
    )
    expected_count = len(expected)

    for dname in TARGET_DIRS:
        dpath = root / dname
        if not dpath.exists():
            print(f"[ERROR] 目录不存在: {dpath}")
            continue

        npz_files = sorted(glob.glob(str(dpath / "*.npz")))
        print(f"\n====== 检查目录: {dname} ======")
        print(f"npz 文件数: {len(npz_files)}")

        expected_prefix = dname.split("_")[0]

        angle_to_files = defaultdict(list)
        foreign_files = []                  # ⚠ 混入其他柿子的文件
        unparsable_files = []               # 文件名解析失败

        for path in npz_files:
            fname = os.path.basename(path)
            info = parse_fname(fname)
            if info is None:
                unparsable_files.append(path)
                continue

            prefix1, roll, pitch = info

            # === 判断是否混入其他柿子 ===
            if prefix1 != expected_prefix:
                foreign_files.append(path)
                continue

            angle_to_files[(roll, pitch)].append(path)

        actual = set(angle_to_files.keys())

        missing = sorted(expected - actual)
        extra = sorted(actual - expected)

        print(f"唯一 (roll,pitch) 组合数 = {len(actual)}（期望: {expected_count}）")
        print(f"缺失组合数量: {len(missing)}")
        if missing:
            for r, p in missing[:20]:
                print(f"  roll={r}, pitch={p}")

        print(f"多出来的组合数量: {len(extra)}")
        if extra:
            for r, p in extra[:20]:
                print(f"  roll={r}, pitch={p}")

        print(f"\n前缀不匹配（疑似混入其他柿子）的文件数量: {len(foreign_files)}")
        if foreign_files:
            print("示例（前 20 个）：")
            for f in foreign_files[:20]:
                print("   -", os.path.basename(f))

            # === ⭐ 添加删除选项 ===========================
            print("\n是否删除这些前缀不匹配的文件？")
            print("  y = 删除全部")
            print("  s = 逐个确认删除")
            print("  n = 不删除")

            choice = input("选择 (y/s/n): ").strip().lower()

            if choice == "y":
                print("\n删除全部前缀不匹配文件中...")
                for f in foreign_files:
                    try:
                        os.remove(f)
                        print(f"删除: {f}")
                    except Exception as e:
                        print(f"删除失败: {f} -> {e}")

            elif choice == "s":
                print("\n逐个确认删除:\n")
                for f in foreign_files:
                    ans = input(f"删除 {f}? (y/n): ").strip().lower()
                    if ans == "y":
                        try:
                            os.remove(f)
                            print(f"删除: {f}")
                        except Exception as e:
                            print(f"删除失败: {f} -> {e}")
                    else:
                        print(f"跳过: {f}")

            else:
                print("不删除前缀不匹配文件。")
            # ============================================

        print(f"\n文件名解析失败的 npz 数量: {len(unparsable_files)}")
        if unparsable_files:
            print("解析失败文件示例：")
            for f in unparsable_files[:20]:
                print("   -", os.path.basename(f))


if __name__ == "__main__":
    main()
