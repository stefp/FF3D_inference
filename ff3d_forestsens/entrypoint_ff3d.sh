#!/usr/bin/env bash
set -euo pipefail

echo "================ FF3D ENTRYPOINT ================"
echo "User: $(whoami)"
echo "Date: $(date)"
echo "PWD at start: $(pwd)"
echo "SKIP_PROVISION=${SKIP_PROVISION:-false}"
echo "SKIP_PREPROCESS=${SKIP_PREPROCESS:-false}"
echo "================================================="

WORK_DIR="/workspace"
SCRIPT="${WORK_DIR}/run_oracle_pipeline.sh"

echo
echo "[1] List /workspace content:"
ls -lah "${WORK_DIR}" || true

echo
echo "[2] Check input/output mount points:"
echo "- Test data (input): ${WORK_DIR}/data/ForAINetV2/test_data"
ls -lah "${WORK_DIR}/data/ForAINetV2/test_data" || true

echo "- Output folder: ${WORK_DIR}/work_dirs/output"
ls -lah "${WORK_DIR}/work_dirs/output" || true

###############################################################################
# 3) In-container fixes and mmengine file replacement (from run_docker_locally)
###############################################################################
if [[ "${SKIP_PROVISION:-false}" == "true" ]]; then
  echo "[3] Provision skipped (SKIP_PROVISION=true)."
else
  echo "[3] Provision inside container (tpk/torch-cluster + file replace)"

  # Find python executable
  PY=$(command -v python || command -v python3)

  echo "[3.1] Check torch_points_kernels import"
  if ! "$PY" - <<'PYCODE'
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

  echo "[3.2] Reinstall torch-cluster"
  pip uninstall -y torch-cluster || true
  pip install --no-deps --no-cache-dir torch-cluster

  echo "[3.3] Replace patched files if present"
  SITE_PKGS=$("$PY" - <<'PYCODE'
import site
c = site.getsitepackages() + [site.getusersitepackages()]
print([p for p in c if p.endswith("site-packages")][0])
PYCODE
)
  REPL_DIR="${WORK_DIR}/replace_mmdetection_files"
  if [[ -d "$REPL_DIR" ]]; then
    [[ -f "$REPL_DIR/loops.py" ]]         && cp "$REPL_DIR/loops.py"        "$SITE_PKGS/mmengine/runner/" || true
    [[ -f "$REPL_DIR/base_model.py" ]]    && cp "$REPL_DIR/base_model.py"   "$SITE_PKGS/mmengine/model/base_model/" || true
    [[ -f "$REPL_DIR/transforms_3d.py" ]] && cp "$REPL_DIR/transforms_3d.py" "$SITE_PKGS/mmdet3d/datasets/transforms/" || true
  fi

  echo "[3.4] Ensure data dirs exist (meta/train_val)"
  mkdir -p "${WORK_DIR}/data/ForAINetV2/meta_data"
  mkdir -p "${WORK_DIR}/data/ForAINetV2/train_val_data"
  # ${WORK_DIR}/data/ForAINetV2/test_data is expected to be mounted from host/OCI

  echo "[3.5] Recheck torch_points_kernels import"
  "$PY" - <<'PYCODE'
from torch_points_kernels import instance_iou
print("torch-points-kernels loaded successfully")
PYCODE
fi

############################################
# 4) laspy preprocessing (optional)
############################################
if [[ "${SKIP_PREPROCESS:-false}" == "true" ]]; then
  echo "[4] Preprocessing skipped (SKIP_PREPROCESS=true)."
else
  echo "[4] Data preprocessing with laspy (install only, scripts optional)"

  # Step 1: install laspy + lazrs
  pip install -q laspy
  pip install -q "laspy[lazrs]"

  # Step 2/3: if you later really want to run these, uncomment:
  # cd "${WORK_DIR}/data/ForAINetV2"
  # python batch_load_ForAINetV2_data.py
  # cd "${WORK_DIR}"
  # python tools/create_data_forainetv2.py forainetv2
fi

############################################
# 5) Kick off pipeline inside container
############################################
echo
echo "[5] Prepare to run pipeline"
if [[ ! -f "${SCRIPT}" ]]; then
  echo "❌ ERROR: run_oracle_pipeline.sh NOT FOUND at:"
  echo "   ${SCRIPT}"
  echo "Searching for it..."
  find "${WORK_DIR}" -maxdepth 4 -name "run_oracle_pipeline.sh" 2>/dev/null || true
  exit 2
fi

echo "[5.1] Script found:"
ls -lah "${SCRIPT}"

echo
echo "[5.2] Change directory to ${WORK_DIR}"
cd "${WORK_DIR}"
echo "PWD now: $(pwd)"

echo "================================================="
echo ">>> STARTING PIPELINE: bash ${SCRIPT}"
echo "================================================="

# Use -ex to show every command executed inside the pipeline
bash -ex "${SCRIPT}"

echo "================================================="
echo ">>> PIPELINE FINISHED SUCCESSFULLY"
echo "================================================="
