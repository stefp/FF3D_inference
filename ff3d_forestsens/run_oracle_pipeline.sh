#!/usr/bin/env bash
set -euo pipefail

############################################
# Basic Path Configuration
############################################
WORK_DIR="${WORK_DIR:-/workspace}"

# Docker mount paths:
# /app/code_binbin/FF3D_oracle/bucket_in_folder  → /workspace/data/ForAINetV2/test_data
# /app/code_binbin/FF3D_oracle/bucket_out_folder → /workspace/work_dirs/output
IN_BUCKET="${IN_BUCKET:-$WORK_DIR/data/ForAINetV2/test_data}"
OUT_BUCKET="${OUT_BUCKET:-$WORK_DIR/work_dirs/output}"

# Inference script
INFER_SH="${INFER_SH:-$WORK_DIR/tools/inference_bluepoint_forestsens.sh}"

# Final subdirectory (produced by the inference script)
FINAL_SUBDIR="${FINAL_SUBDIR:-round_2_after_remove_noise_200}"

# Execution options
CLEAR_INPUT_AFTER_RUN="${CLEAR_INPUT_AFTER_RUN:-true}"
CLEAR_OUTPUT_BEFORE_RUN="${CLEAR_OUTPUT_BEFORE_RUN:-true}"
KEEP_ONLY_ZIP="${KEEP_ONLY_ZIP:-true}"
RESTORE_ORIGINAL_NAMES="${RESTORE_ORIGINAL_NAMES:-false}"

# Metadata for restoring original names
META_DIR="${META_DIR:-$WORK_DIR/data/ForAINetV2/meta_data}"
SAFE_LIST="$META_DIR/test_list_initial.txt"
ORIG_LIST="$META_DIR/test_list_initial_original.txt"

log(){ echo "[$(date +%H:%M:%S)] $*"; }
safe_mkdir(){ mkdir -p "$@"; }
purge_dir(){ local d="$1"; [[ -n "$d" && "$d" != "/" ]] && rm -rf "${d:?}/"*; }

############################################
# Print Path Information
############################################
log "IN_BUCKET   = $IN_BUCKET"
log "OUT_BUCKET  = $OUT_BUCKET"
log "FINAL_SUBDIR= $FINAL_SUBDIR"
log "CLEAR_INPUT_AFTER_RUN=$CLEAR_INPUT_AFTER_RUN, KEEP_ONLY_ZIP=$KEEP_ONLY_ZIP, RESTORE_ORIGINAL_NAMES=$RESTORE_ORIGINAL_NAMES"

safe_mkdir "$IN_BUCKET" "$OUT_BUCKET"

############################################
# Step 0: Clean output directory before run
############################################
if [[ "$CLEAR_OUTPUT_BEFORE_RUN" == "true" ]]; then
  log "Clearing output directory: $OUT_BUCKET"
  purge_dir "$OUT_BUCKET"
fi

############################################
# Step 1: Unzip input files (if any) — Python (no system 'unzip' needed)
############################################
python3 - <<'PYCODE'
import os, sys, zipfile
in_bucket = os.environ.get("IN_BUCKET", "/workspace/data/ForAINetV2/test_data")
if not os.path.isdir(in_bucket):
    sys.exit(0)

for fn in sorted(os.listdir(in_bucket)):
    if not fn.lower().endswith(".zip"):
        continue
    zpath = os.path.join(in_bucket, fn)
    print(f"Unzipping: {fn}")
    try:
        with zipfile.ZipFile(zpath, 'r') as zf:
            zf.extractall(in_bucket)
        os.remove(zpath)
    except Exception as e:
        print(f"[ERROR] Failed to unzip {fn}: {e}", file=sys.stderr)
PYCODE

############################################
# Step 2: Convert .las / .laz → .ply
############################################
COUNT_LAS=$(find "$IN_BUCKET" -maxdepth 1 -type f \( -iname "*.las" -o -iname "*.laz" \) | wc -l)
if (( COUNT_LAS > 0 )); then
  log "Detected $COUNT_LAS LAS/LAZ files, converting to PLY..."
  python3 - <<'PYCODE'
import os, laspy, numpy as np
from plyfile import PlyData, PlyElement
from glob import glob

input_folder = os.environ.get("IN_BUCKET", "/workspace/data/ForAINetV2/test_data")
laz_files = glob(os.path.join(input_folder, "*.laz")) + glob(os.path.join(input_folder, "*.las"))

for laz_file in laz_files:
    basename = os.path.splitext(os.path.basename(laz_file))[0]
    ply_file = os.path.join(input_folder, f"{basename}.ply")
    print(f"Converting {laz_file} → {ply_file}")
    try:
        with laspy.open(laz_file) as f:
            las = f.read()
        pts = np.vstack((las.x, las.y, las.z)).astype(np.float64).T

        # Temporary normalization for numerical stability (reverted immediately)
        mean_x, mean_y, min_z = np.mean(pts[:, 0]), np.mean(pts[:, 1]), np.min(pts[:, 2])
        pts[:, 0] -= mean_x; pts[:, 1] -= mean_y; pts[:, 2] -= min_z
        pts[:, 0] += mean_x; pts[:, 1] += mean_y; pts[:, 2] += min_z

        vertex = np.array([tuple(row) for row in pts],
                          dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8')])
        PlyData([PlyElement.describe(vertex, 'vertex')], text=False).write(ply_file)
    except Exception as e:
        print(f"[ERROR] Failed to convert {laz_file}: {e}")

# Remove original LAS/LAZ files
for ext in ("*.laz", "*.las"):
    for f in glob(os.path.join(input_folder, ext)):
        os.remove(f)
PYCODE
  log "LAS/LAZ conversion completed ✅"
fi

############################################
# Step 3: Run inference (rename in place)
############################################
log "Starting inference (in-place renaming on IN_BUCKET)..."
TEST_DATA_DIR="$IN_BUCKET" bash "$INFER_SH"
log "Inference finished."

############################################
# Step 4: Collect and export final results
############################################
FINAL_DIR="$OUT_BUCKET/$FINAL_SUBDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
ZIP_NAME="results_${STAMP}.zip"
ZIP_PATH="/tmp/${ZIP_NAME}"

if [[ -d "$FINAL_DIR" ]]; then
  log "Exporting final results: $FINAL_DIR → $OUT_BUCKET"
  STAGE="/tmp/final_stage_$$"
  safe_mkdir "$STAGE"
  rsync -a "$FINAL_DIR"/ "$STAGE"/
  purge_dir "$OUT_BUCKET"
  rsync -a "$STAGE"/ "$OUT_BUCKET"/
  rm -rf "$STAGE"

  ##########################################
  # Optional: restore original names
  ##########################################
  if [[ "$RESTORE_ORIGINAL_NAMES" == "true" && -f "$SAFE_LIST" && -f "$ORIG_LIST" ]]; then
    log "Restoring original names (before renaming)..."
    python3 - <<'PYCODE'
import os, re
out_dir   = os.environ.get("OUT_BUCKET")
safe_list = os.environ.get("SAFE_LIST")
orig_list = os.environ.get("ORIG_LIST")

def read_lines(p):
    with open(p, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]

safe = read_lines(safe_list)
orig = read_lines(orig_list)
pairs = sorted(zip(safe, orig), key=lambda x: len(x[0]), reverse=True)

round_suffix_re = re.compile(r"^(?P<stem>.+?)(?P<round>_round\d+)(?P<ext>\.[^.]+)?$", re.IGNORECASE)

for fn in os.listdir(out_dir):
    old_path = os.path.join(out_dir, fn)
    if not os.path.isfile(old_path):
        continue

    name, ext = os.path.splitext(fn)
    m = round_suffix_re.match(fn)
    if m:
        stem   = m.group("stem")
        roundt = m.group("round")
        ext    = m.group("ext") or ""
    else:
        stem   = name
        roundt = ""

    new_stem = stem
    for s, o in pairs:
        if new_stem.startswith(s):
            new_stem = o + new_stem[len(s):]
            break

    new_fn = f"{new_stem}{roundt}{ext}"
    new_path = os.path.join(out_dir, new_fn)
    if new_path != old_path and not os.path.exists(new_path):
        os.rename(old_path, new_path)
PYCODE
  fi

  ##########################################
  # Pack results into a ZIP file (Python)
  ##########################################
  log "Packing results into ZIP: $ZIP_PATH"
  export OUT_BUCKET ZIP_PATH
  python3 - <<'PYCODE'
import os, zipfile
out_dir = os.environ['OUT_BUCKET']
zip_path = os.environ['ZIP_PATH']
if os.path.exists(zip_path):
    os.remove(zip_path)
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in os.listdir(out_dir):
        fp = os.path.join(out_dir, f)
        if os.path.isfile(fp) and not f.endswith(".zip"):
            zf.write(fp, f)
print(f"✔ Created ZIP → {zip_path}")
PYCODE

  ##########################################
  # Optionally keep only the ZIP file
  ##########################################
  if [[ "$KEEP_ONLY_ZIP" == "true" ]]; then
    log "Keeping only ZIP file: clearing $OUT_BUCKET and placing ZIP back"
    purge_dir "$OUT_BUCKET"
    mv -f "$ZIP_PATH" "$OUT_BUCKET/"
  else
    mv -f "$ZIP_PATH" "$OUT_BUCKET/"
  fi
else
  log "[WARNING] Final directory not found: $FINAL_DIR"
fi

############################################
# Step 5: Clear input folder (optional)
############################################
if [[ "$CLEAR_INPUT_AFTER_RUN" == "true" ]]; then
  log "Clearing input folder: $IN_BUCKET (only top-level files)"
  find "$IN_BUCKET" -maxdepth 1 -type f -delete
fi

log "Done ✅  Results have been generated in: $OUT_BUCKET"
