#!/usr/bin/env bash
set -euo pipefail

############################################
# Config (overridable via environment variables)
############################################
IMAGE_NAME="${IMAGE_NAME:-forestformer-forestsens-image}"
CONTAINER_NAME="${CONTAINER_NAME:-forestformer-forestsens-container}"

# Your project directory (must contain run_oracle_pipeline.sh, tools, replace_mmdetection_files, etc.)
HOST_PROJECT_DIR="${HOST_PROJECT_DIR:-$(pwd)}"

# Bucket in/out on the host
HOST_BUCKET_IN="${HOST_BUCKET_IN:-/app/code_binbin/FF3D_oracle/bucket_in_folder}"
HOST_BUCKET_OUT="${HOST_BUCKET_OUT:-/app/code_binbin/FF3D_oracle/bucket_out_folder}"

# Mount targets inside the container (aligned with code/README)
CONTAINER_TEST_DATA="/workspace/data/ForAINetV2/test_data"
CONTAINER_OUTPUT_DIR="/workspace/work_dirs/output"

# Resources and ports
SHM_SIZE="${SHM_SIZE:-128g}"
HOST_SSH_PORT="${HOST_SSH_PORT:-49218}"

# Build/Run policies: auto|always|never
REBUILD_IMAGE="${REBUILD_IMAGE:-auto}"
RECREATE_CONTAINER="${RECREATE_CONTAINER:-auto}"

# Optional skips
SKIP_PROVISION="${SKIP_PROVISION:-false}"   # skip tpk/torch-cluster fix + file replacement
SKIP_PREPROCESS="${SKIP_PREPROCESS:-false}" # skip laspy preprocessing (batch_load + create_data)

printf "Current directory: %s\n" "$(pwd)"

############################################
# 0) Pre-check
############################################
docker info >/dev/null 2>&1 || { echo "Docker unavailable"; exit 1; }

mkdir -p "${HOST_BUCKET_IN}" "${HOST_BUCKET_OUT}"

############################################
# 1) Build (on demand)
############################################
echo "[1/5] Build image: ${IMAGE_NAME}"
image_exists="$(docker images -q "${IMAGE_NAME}" 2>/dev/null || true)"

should_build=false
case "$REBUILD_IMAGE" in
  always) should_build=true ;;
  never)  should_build=false ;;
  auto)   [[ -z "$image_exists" ]] && should_build=true || should_build=false ;;
  *) echo "Unknown REBUILD_IMAGE=${REBUILD_IMAGE} (use auto|always|never)"; exit 1 ;;
esac

if $should_build; then
  docker build -t "${IMAGE_NAME}" .
else
  echo "Image exists, skip build. (REBUILD_IMAGE=${REBUILD_IMAGE})"
fi

############################################
# 2) Ensure container + mount buckets (on demand)
############################################
echo "[2/5] Ensure container: ${CONTAINER_NAME}"

run_container () {
  docker run -d \
    --gpus all \
    --shm-size="${SHM_SIZE}" \
    -p "127.0.0.1:${HOST_SSH_PORT}:22" \
    --name "${CONTAINER_NAME}" \
    -v "${HOST_PROJECT_DIR}:/workspace" \
    --mount "type=bind,source=${HOST_BUCKET_IN},target=${CONTAINER_TEST_DATA}" \
    --mount "type=bind,source=${HOST_BUCKET_OUT},target=${CONTAINER_OUTPUT_DIR}" \
    --entrypoint bash \
    "${IMAGE_NAME}" -lc "sleep infinity"
}

container_exists="$(docker ps -a --format '{{.Names}}' | grep -x "${CONTAINER_NAME}" || true)"
container_running="$(docker ps --format '{{.Names}}' | grep -x "${CONTAINER_NAME}" || true)"

case "$RECREATE_CONTAINER" in
  always)
    [[ -n "$container_exists" ]] && docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    run_container
    ;;
  never)
    if [[ -z "$container_exists" ]]; then
      echo "Container not found and RECREATE_CONTAINER=never → creating once."
      run_container
    else
      [[ -z "$container_running" ]] && docker start "${CONTAINER_NAME}" >/dev/null
    fi
    ;;
  auto)
    if [[ -z "$container_exists" ]]; then
      run_container
    else
      [[ -z "$container_running" ]] && docker start "${CONTAINER_NAME}" >/dev/null
    fi
    ;;
  *) echo "Unknown RECREATE_CONTAINER=${RECREATE_CONTAINER} (use auto|always|never)"; exit 1 ;;
esac

echo "[2/5] Container is up (kept alive with sleep infinity)."

############################################
# 3) In-container fixes and mmengine file replacement (optional)
############################################
if [[ "$SKIP_PROVISION" == "true" ]]; then
  echo "[3/5] Provision skipped (SKIP_PROVISION=true)."
else
  echo "[3/5] Provision inside container (tpk/torch-cluster + file replace)"
  docker exec -i "${CONTAINER_NAME}" bash -lc '
set -euo pipefail

# Find python executable
PY=$(command -v python || command -v python3)

echo "[Step 2] Check torch_points_kernels import"
if ! "$PY" - <<'"PYCODE"'
try:
    from torch_points_kernels import instance_iou
    print("tpk ok")
except Exception as e:
    print("tpk import failed:", e)
    raise SystemExit(1)
PYCODE
then
  echo "  -> reinstall torch-points-kernels==0.7.0"
  pip uninstall -y torch-points-kernels || true
  pip install --no-deps --no-cache-dir torch-points-kernels==0.7.0
fi

echo "[Step 3] Reinstall torch-cluster"
pip uninstall -y torch-cluster || true
pip install --no-deps --no-cache-dir torch-cluster

echo "[Step 4] Replace patched files if present"
SITE_PKGS=$("$PY" - <<'"PYCODE"'
import site
c = site.getsitepackages() + [site.getusersitepackages()]
print([p for p in c if p.endswith("site-packages")][0])
PYCODE
)
REPL_DIR="/workspace/replace_mmdetection_files"
if [[ -d "$REPL_DIR" ]]; then
  [[ -f "$REPL_DIR/loops.py" ]]        && cp "$REPL_DIR/loops.py"        "$SITE_PKGS/mmengine/runner/" || true
  [[ -f "$REPL_DIR/base_model.py" ]]   && cp "$REPL_DIR/base_model.py"   "$SITE_PKGS/mmengine/model/base_model/" || true
  [[ -f "$REPL_DIR/transforms_3d.py" ]]&& cp "$REPL_DIR/transforms_3d.py" "$SITE_PKGS/mmdet3d/datasets/transforms/" || true
fi

echo "[Dirs] Ensure data dirs exist"
mkdir -p /workspace/data/ForAINetV2/meta_data
mkdir -p /workspace/data/ForAINetV2/train_val_data
# /workspace/data/ForAINetV2/test_data is provided by the bucket bind mount

echo "[Recheck] torch_points_kernels import"
"$PY" - <<'"PYCODE"'
from torch_points_kernels import instance_iou
print("torch-points-kernels loaded successfully")
PYCODE
'
fi

############################################
# 4) laspy preprocessing (optional)
############################################
if [[ "$SKIP_PREPROCESS" == "true" ]]; then
  echo "[4/5] Preprocessing skipped (SKIP_PREPROCESS=true)."
else
  echo "[4/5] Data preprocessing with laspy"
  docker exec -i "${CONTAINER_NAME}" bash -lc '
set -euo pipefail
# cd /workspace/data/ForAINetV2

# Step 1: install laspy + lazrs
pip install -q laspy
pip install -q "laspy[lazrs]"

# Step 2: run data loader script (generates forainetv2_instance_data)
# python batch_load_ForAINetV2_data.py

# Step 3: back to project root and create training data (if needed)
# cd /workspace
# python tools/create_data_forainetv2.py forainetv2
'
fi

############################################
# 5) Kick off pipeline inside container
############################################
echo "[5/5] Kick off pipeline inside container"
echo "IN  → ${CONTAINER_TEST_DATA}"
echo "OUT → ${CONTAINER_OUTPUT_DIR}"
docker exec -i "${CONTAINER_NAME}" bash -lc "
set -euo pipefail
bash /workspace/run_oracle_pipeline.sh
"
echo "✅ Pipeline finished."

echo
echo "✅ All done."
echo "Logs:     docker logs -f ${CONTAINER_NAME}"
echo "Shell:    docker exec -it ${CONTAINER_NAME} /bin/bash"
echo
echo "Training example (inside container):"
echo "  export PYTHONPATH=/workspace"
echo "  CUDA_VISIBLE_DEVICES=0 python tools/train.py /workspace/configs/oneformer3d_qs_radius16_qp300_2many.py --work-dir /workspace/work_dirs/<output_folder_name>"
echo
echo "Testing example (inside container):"
echo "  python tools/fix_spconv_checkpoint.py --in-path work_dirs/oneformer3d_1xb4_forainetv2/trained.pth --out-path work_dirs/oneformer3d_1xb4_forainetv2/trained_fix.pth"
echo "  # after modifying output_path in predict():"
echo "  CUDA_VISIBLE_DEVICES=0 python tools/test.py /workspace/configs/oneformer3d_qs_radius16_qp300_2many.py work_dirs/oneformer3d_1xb4_forainetv2/trained_fix.pth"
