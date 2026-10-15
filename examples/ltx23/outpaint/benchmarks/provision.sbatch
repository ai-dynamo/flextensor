#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=ltx_provision
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --time=06:00:00
set -x
mkdir -p $BASE/{inputs,out,hf} || exit 1
export HF_HOME=$BASE/hf HF_HUB_CACHE=$BASE/hf/hub HF_XET_CACHE=$BASE/hf/xet
# xtrace prints expanded values, and this job's log lands on shared storage. Disable it across the
# token handling so the secret is never written to the Slurm log, then restore it.
set +x
export HF_TOKEN=${HF_TOKEN:?export your Hugging Face token first (gemma-3-12b is gated)}
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
set -x
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HUB_CACHE" "$HF_XET_CACHE"
hostname
nvidia-smi -L
nvidia-smi topo -m | head -10
df -h /raid | tail -1
free -g | head -2

# uv (node-local)
export PATH="$BASE/bin:$PATH"
if ! command -v uv >/dev/null; then
  export UV_INSTALL_DIR=$BASE/bin
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$BASE/bin sh
fi
export PATH="$BASE/bin:$PATH"
uv --version || exit 1

# LTX-2
# LTX-2 is pinned. Cloning its default branch made the benchmark silently
# unreproducible: upstream replaced ICLoraPipeline's keyword arguments with a ModelPaths object on
# 2026-08-11 (commit 2362161), so a later clone fails with
#   TypeError: ICLoraPipeline.__init__() got an unexpected keyword argument 'distilled_checkpoint_path'
# 4f89057 (2026-08-03) is the last commit with the signature this example targets, and is what the
# published numbers were measured against. Override with LTX2_COMMIT to test a newer upstream.
LTX2_COMMIT=${LTX2_COMMIT:-4f8905737aac86a554637cac86c178877a39c744}
[ -d $BASE/LTX-2 ] || {
  git clone -q --filter=blob:none https://github.com/Lightricks/LTX-2.git $BASE/LTX-2
  git -C $BASE/LTX-2 checkout -q "$LTX2_COMMIT"
}
cd $BASE/LTX-2
export UV_PROJECT_ENVIRONMENT=$BASE/LTX-2/.venv UV_LINK_MODE=copy UV_CACHE_DIR=$BASE/uvcache
uv sync >$BASE/uv_sync.log 2>&1
echo "uv_sync_rc=$?"
PY=$BASE/LTX-2/.venv/bin/python
uv pip install --python $PY -q flextensor huggingface_hub hf_transfer 2>&1 | tail -2
$PY -c "import torch,ltx_pipelines,flextensor;print('VENV_OK torch',torch.__version__,'gpus',torch.cuda.device_count())"

# weights
$PY - <<'PYEOF'
import os
from huggingface_hub import hf_hub_download, snapshot_download
c=os.environ["HF_HUB_CACHE"]
snapshot_download("google/gemma-3-12b-it-qat-q4_0-unquantized",cache_dir=c); print("GEMMA_OK",flush=True)
for r,f in [("oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint","ltx-2.3-22b-ic-lora-outpaint.safetensors"),
            ("Lightricks/LTX-2.3","ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
            ("Lightricks/LTX-2.3","ltx-2.3-22b-distilled-1.1.safetensors")]:
    hf_hub_download(r,f,cache_dir=c); print("OK",r,f,flush=True)
print("WEIGHTS_READY")
PYEOF

# scripts + inputs + the 30s conditioning clip
cp "$LTX_SRC"/*.py $BASE/
cp $STAGE/inputs/*.mp4 $BASE/inputs/
# -c copy: re-encoding bleeds the inner content into the letterbox bars, and the Outpaint
# IC-LoRA fills regions that are *exactly* RGB(0,0,0).
command -v ffmpeg >/dev/null && ffmpeg -y -loglevel error -stream_loop 4 -t 30 -i $BASE/inputs/real_input_8s.mp4 -c copy $BASE/inputs/real_input_30s.mp4 ||
  $PY - <<'PYEOF'
# no ffmpeg on the node: build the 30s loop with PyAV (ships with LTX-2)
import av, os
base=os.environ.get("BASE","/tmp/ltx_pk")
src=f"{base}/inputs/real_input_8s.mp4"; dst=f"{base}/inputs/real_input_30s.mp4"
inp=av.open(src); vs=inp.streams.video[0]
out=av.open(dst,"w"); os_=out.add_stream("libx264", rate=vs.average_rate)
os_.width, os_.height, os_.pix_fmt = vs.codec_context.width, vs.codec_context.height, "yuv444p"
os_.options={"crf":"0"}  # lossless: keep the letterbox bars exactly black
n=0
while n < 720:
    inp2=av.open(src)
    for fr in inp2.decode(video=0):
        if n>=720: break
        for p in os_.encode(fr): out.mux(p)
        n+=1
    inp2.close()
for p in os_.encode(): out.mux(p)
out.close(); print("BUILT_30S frames", n)
PYEOF
ls -la $BASE/inputs/
echo "PROVISION_DONE"
