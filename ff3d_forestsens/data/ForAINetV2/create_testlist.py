import os
from glob import glob

ply_dir = "/workspace/data/ForAINetV2/test_data"
output_txt = "/workspace/data/ForAINetV2/meta_data/test_list.txt"

# 获取所有 .ply 文件路径
ply_files = glob(os.path.join(ply_dir, "*.ply"))

# 按文件大小从小到大排序
ply_files.sort(key=lambda f: os.path.getsize(f))

# 获取不带后缀的文件名
base_names = [os.path.splitext(os.path.basename(f))[0] for f in ply_files]

# 写入到 test_list.txt
with open(output_txt, "w") as f:
    for name in base_names:
        f.write(name + "\n")

print(f"Written {len(base_names)} entries to {output_txt}, sorted by file size (ascending).")
