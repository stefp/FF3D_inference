# ForestFormer3D-Forestsens — Inference Guide

## 0. Folder Structure

```
/YOUR/PATH/
├── ff3d_forestsens/
│   ├── run_docker_locally.sh
│   ├── entrypoint_ff3d.sh
│   ├── tools/
│   │   └── inference_bluepoint_forestsens.sh
│   ├── Dockerfile
│   ├── configs/...
│   └── ...
│
└── FF3D_oracle/
    ├── bucket_in_folder/      # Input (auto-cleared)
    └── bucket_out_folder/     # Output (ZIP)
```

## 1. Hardware & GPU Requirements

- Requires a **high-memory GPU**, preferably **NVIDIA A100**.
- Only **single-GPU** inference is supported.
- If you still get **CUDA out of memory**, reduce the `chunk` size in the config:

Edit:

```bash
/YOUR/PATH/ff3d_forestsens/configs/oneformer3d_qs_radius16_qp300_2many.py
```

Find the `chunk` parameter and set it to a **smaller value**.

---

---

## 2. Path Setup (IMPORTANT)

This documentation uses the example path:

```
/app/code_binbin/
```

Please **replace it with your own actual project path** inside file run_docker_locally.sh.

```bash
# Bucket in/out on the host
HOST_BUCKET_IN="${HOST_BUCKET_IN:-/app/code_binbin/FF3D_oracle/bucket_in_folder}"
HOST_BUCKET_OUT="${HOST_BUCKET_OUT:-/app/code_binbin/FF3D_oracle/bucket_out_folder}"

to

HOST_BUCKET_IN="${HOST_BUCKET_IN:-/YOUR/PATH/FF3D_oracle/bucket_in_folder}"
HOST_BUCKET_OUT="${HOST_BUCKET_OUT:-/YOUR/PATH/FF3D_oracle/bucket_out_folder}"
```

---

## 3. Project Location

Navigate to the project directory:

```bash
cd /YOUR/PATH/ff3d_forestsens
```

---

## 4. Prepare Input Files

Place your test files into:

```
/YOUR/PATH/FF3D_oracle/bucket_in_folder
```

Supported formats:
- `.ply`
- `.las` / `.laz`
- `.zip` containing `.ply` or `.laz`

⚠️ **The input folder will be automatically cleared after inference.  
Backup your files before running.**

---

## 5. Run Inference

Execute:

```bash
sudo bash run_docker_locally.sh
```

---

## 6. Output Results

Results are saved to:

```
/YOUR/PATH/FF3D_oracle/bucket_out_folder
```

Output is a single **.zip** file.

---

## 7. Optional Settings

### Change GPU
Edit:

```
/YOUR/PATH/ff3d_forestsens/tools/inference_bluepoint_forestsens.sh
```

Modify:

```bash
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
```

### Change Number of Iterations

In the same file:

```bash
ITERATIONS="${ITERATIONS:-2}"
```

Set to 1 / 2 / 3+ as needed.

---

## 8. Quick Summary

```bash
cd /YOUR/PATH/ff3d_forestsens
cp your_files /YOUR/PATH/FF3D_oracle/bucket_in_folder/
sudo bash run_docker_locally.sh

# Output:
# /YOUR/PATH/FF3D_oracle/bucket_out_folder/output.zip
```
