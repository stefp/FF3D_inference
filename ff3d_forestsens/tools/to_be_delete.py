import os

ply_dir = "/workspace/work_dirs/bluepoint_th04fixed_03_priority_test_merged/round_1"
txt_file = "/workspace/data/ForAINetV2/meta_data/test_list_initial_original.txt"


with open(txt_file, 'r') as f:
    prefixes = [line.strip() for line in f if line.strip()]

ply_files = os.listdir(ply_dir)

unprocessed = []
for prefix in prefixes:
    if not any(prefix in ply_name for ply_name in ply_files):
        unprocessed.append(prefix)

print(f"There are {len(prefixes)} file prefixes in total.")
print(f"{len(unprocessed)} of them have not been processed (no matching .ply file found):\n")

for name in unprocessed:
    print(name)
