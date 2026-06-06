import os
import subprocess
import sys

VENV_PY = os.path.join(os.path.dirname(__file__), "venv", "lib", "python3.13", "site-packages")
CUDA_LIBS = [
    "nvidia/cuda_runtime/lib",
    "nvidia/cublas/lib",
    "nvidia/cudnn/lib",
    "nvidia/cufft/lib",
    "nvidia/cusolver/lib",
    "nvidia/cusparse/lib",
    "nvidia/nvjitlink/lib",
]
cuda_paths = [os.path.join(VENV_PY, p) for p in CUDA_LIBS]
os.environ["LD_LIBRARY_PATH"] = ":".join(cuda_paths) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={os.path.join(VENV_PY, 'nvidia', 'cuda_nvcc')}"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

def run_script(command, step_name):
    print(f"\n{'='*50}\n[STARTING STEP] {step_name}\n{'='*50}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"\n[ERROR] Pipeline failed at {step_name}. Exiting.")
        sys.exit(1)
    print(f"\n[SUCCESS] {step_name} completed successfully.")

if __name__ == "__main__":
    print("=== MULTI-TASK PIPELINE ORCHESTRATOR ===")

    # JS data extraction from Mongo GridFS -> Local
    run_script("node scripts/dataloader.js", "JS Data Extraction")

    # Python preprocess -> New Mongo GridFS
    run_script("python scripts/processor.py", "Python Preprocessing")

    # Python training from New Mongo GridFS
    run_script("python scripts/train.py", "Model Training")

    # Python converting to JS model
    run_script("python scripts/export.py", "TFJS Export")

    print("\n=== PIPELINE FINISHED COMPLETELY ===")
    print("Your final model is ready for your web application!")