"""Kaggle script kernel: train Ouroboros v2 candidates on the GPU, one after another, and verify each.

Code and corpus come from the private dataset `ouroboros-code` (see kaggle/stage.sh); nothing is
installed.  Outputs land in /kaggle/working/runs/<name>/{ckpt.pt,log.json,verify.json}.
"""
import os
import shutil
import subprocess
import sys
import time

SRC = "/kaggle/input/ouroboros-code"
W = "/kaggle/working"
if os.path.isdir(f"{SRC}/ouroboros"):
    shutil.copytree(f"{SRC}/ouroboros", f"{W}/ouroboros", dirs_exist_ok=True)
else:  # the CLI uploads folders as archives
    import zipfile
    zipfile.ZipFile(f"{SRC}/ouroboros.zip").extractall(W if zipfile.ZipFile(f"{SRC}/ouroboros.zip").namelist()[0].startswith("ouroboros/") else f"{W}/ouroboros")
print("code:", sorted(os.listdir(f"{W}/ouroboros")))
os.makedirs(f"{W}/data.noindex", exist_ok=True)
shutil.copy(f"{SRC}/shakespeare.txt", f"{W}/data.noindex/shakespeare.txt")
os.chdir(W)
import torch  # noqa: E402
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

ARCH = ["--d", "128", "--n_layer", "4", "--n_head", "8", "--block", "256", "--seq", "256",
        "--batch", "64", "--qbatch", "1024", "--warmup", "300", "--eval_every", "500", "--eval_n", "16", "--threads", "2"]
CIRCUIT = ["--init_circuit", "--protect", "0.0", "--scale", "1.0"]
RUNS = [
    ("c_lr3e-3", CIRCUIT + ["--lr", "3e-3", "--steps", "10000"]),
    ("c_lr1e-2", CIRCUIT + ["--lr", "1e-2", "--steps", "10000"]),
    ("c_lr1e-2_fd02", CIRCUIT + ["--lr", "1e-2", "--steps", "10000", "--free_density", "0.2"]),
    ("c_lr2e-2", CIRCUIT + ["--lr", "2e-2", "--steps", "10000"]),
    ("c_lr1e-2_lam0", CIRCUIT + ["--lr", "1e-2", "--steps", "10000", "--lam", "0.0"]),
    ("plain_s1_lr1e-2", ["--strict_causal", "--scale", "1.0", "--lr", "1e-2", "--steps", "10000"]),
]
for name, extra in RUNS:
    t0 = time.time()
    print(f"\n===== {name} =====", flush=True)
    subprocess.run([sys.executable, "-m", "ouroboros.train_tm", "--out", f"runs/{name}", *ARCH, *extra], check=True)
    subprocess.run([sys.executable, "-m", "ouroboros.verify_tm", f"runs/{name}/ckpt.pt", "--generations", "1",
                    "--device", "cuda", "--eval_n", "16", "--threads", "2"], check=False)
    print(f"===== {name} done in {time.time() - t0:.0f}s", flush=True)
