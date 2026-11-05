"""Kaggle script kernel: train Ouroboros v2 candidates on the GPU, one after another, and verify each.

Code and corpus come from the private dataset `ouroboros-code` (see kaggle/stage.sh); nothing is
installed.  Outputs land in /kaggle/working/runs/<name>/{ckpt.pt,log.json,verify.json}.
"""
import os
import shutil
import subprocess
import sys
import time

SRC = "/kaggle/input"  # the dataset is mounted somewhere under here; search for the code
W = "/kaggle/working"
for root, dirs, files in os.walk(SRC):
    print("input:", root, sorted(dirs), sorted(files)[:12], file=sys.stderr, flush=True)
cands = [r for r, _, f in os.walk(SRC) if "tm.py" in f and "circuit.py" in f]
if cands:
    shutil.copytree(cands[0], f"{W}/ouroboros", dirs_exist_ok=True)
else:  # the CLI may upload a folder as an archive
    import zipfile
    z = [os.path.join(r, f) for r, _, fs in os.walk(SRC) for f in fs if f.endswith(".zip")][0]
    zipfile.ZipFile(z).extractall(f"{W}/zip")
    src = [r for r, _, f in os.walk(f"{W}/zip") if "tm.py" in f][0]
    shutil.copytree(src, f"{W}/ouroboros", dirs_exist_ok=True)
print("code:", sorted(os.listdir(f"{W}/ouroboros")))
os.makedirs(f"{W}/data.noindex", exist_ok=True)
shutil.copy([os.path.join(r, f) for r, _, fs in os.walk(SRC) for f in fs if f == "shakespeare.txt"][0], f"{W}/data.noindex/shakespeare.txt")
os.chdir(W)
import torch  # noqa: E402
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

ARCH = ["--d", "128", "--n_layer", "4", "--n_head", "8", "--block", "256", "--seq", "256",
        "--batch", "64", "--qbatch", "1024", "--warmup", "300", "--eval_every", "500", "--eval_n", "16", "--threads", "2"]
CIRCUIT = ["--init_circuit", "--protect", "0.0", "--n_scale", "2", "--scale", "1.0"]
# Two scale groups: the circuit rows stay at 1.0, the free rows train at a small scale (what the
# language model wants).  Sweep the free scale and the learning rate; one plain baseline for the ceiling.
RUNS = [
    ("c2_fs015_lr3e-3", CIRCUIT + ["--free_scale", "0.15", "--lr", "3e-3", "--steps", "10000"]),
    ("plain_s015_lr3e-3", ["--strict_causal", "--scale", "0.15", "--lam", "0.0", "--lr", "3e-3", "--steps", "10000"]),
    ("c2_fs015_lr1e-3", CIRCUIT + ["--free_scale", "0.15", "--lr", "1e-3", "--steps", "10000"]),
    ("c2_fs025_lr3e-3", CIRCUIT + ["--free_scale", "0.25", "--lr", "3e-3", "--steps", "10000"]),
]
for name, extra in RUNS:
    t0 = time.time()
    print(f"\n===== {name} =====", flush=True)
    subprocess.run([sys.executable, "-m", "ouroboros.train_tm", "--out", f"runs/{name}", *ARCH, *extra], check=True)
    subprocess.run([sys.executable, "-m", "ouroboros.verify_tm", f"runs/{name}/ckpt.pt", "--generations", "3",
                    "--device", "cuda", "--eval_n", "16", "--threads", "2"], check=False)
    print(f"===== {name} done in {time.time() - t0:.0f}s", flush=True)
