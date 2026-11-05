#!/bin/sh
# Assemble the dataset folder Kaggle kernels read the code from: ouroboros/*.py + the text corpus.
# Usage: kaggle/stage.sh <staging-dir>   then   kaggle datasets create -p <dir> --dir-mode zip   (first time)
#                                       or     kaggle datasets version -p <dir> --dir-mode zip -m "msg"
set -e
cd "$(dirname "$0")/.."
D="$1"; mkdir -p "$D/ouroboros"
cp ouroboros/*.py "$D/ouroboros/"
cp data.noindex/shakespeare.txt "$D/"
cp kaggle/dataset-metadata.json "$D/"
ls -la "$D" "$D/ouroboros"
