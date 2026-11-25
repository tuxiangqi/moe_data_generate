# data_generate

用上游 vLLM 推理并采集 MoE 路由数据。采集、重复终止和停止元数据由本仓库通过 vLLM 的 `worker_cls` 和 `scheduler_cls` 接入，强制 token 回放由上游 vLLM 提供。部署只需要本仓库、官方 wheel、模型和输入数据。裸机与 Slurm 共用同一套安装与配置。

兼容基准为 **vLLM 0.32.0 + CUDA 12.9（cu129）+ Python 3.12**。此版本使用 GPU Model Runner V2、eager 和单节点 tensor parallel。端到端 GPU 验收待完成。

## 安装 uv 环境

从仓库根目录运行。示例适用于 Linux x86_64（glibc 2.28 或更新版）。用 `nvidia-smi` 检查驱动与可用 GPU，建议使用原生支持 CUDA 12.9 的驱动。预编译 wheel 安装无需编译 vLLM。[vLLM 0.32.0 GPU 安装文档](https://docs.vllm.ai/en/v0.32.0/getting_started/installation/gpu/)

如果尚未安装 uv，按 [uv 官方安装说明](https://docs.astral.sh/uv/getting-started/installation/) 安装。然后执行：

```bash
uv --version
uv python install 3.12
uv venv --python 3.12 .venv

uv pip install --python .venv/bin/python \
  'vllm==0.32.0+cu129' \
  --extra-index-url https://wheels.vllm.ai/0.32.0/cu129 \
  --torch-backend=cu129

uv pip check --python .venv/bin/python
```

仓库脚本会将本项目的 `src/` 加到 Python 搜索路径。使用干净的新环境，避免继承指向其他 vLLM 源码树的 `PYTHONPATH`。在有 GPU 的机器或 Slurm GPU allocation 内检查版本和 CUDA：

```bash
nvidia-smi
.venv/bin/python - <<'PY'
from importlib.metadata import version
import torch
import vllm

print("vLLM:", version("vllm"), vllm.__file__)
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
assert version("vllm") == "0.32.0+cu129"
assert torch.version.cuda == "12.9"
assert torch.cuda.is_available(), "CUDA unavailable"
for i in range(torch.cuda.device_count()):
    info = torch.cuda.get_device_properties(i)
    print(i, info.name, f"{info.total_memory / 2**30:.1f} GiB")
PY
```

## 准备模型和本机配置

使用与原采集一致的 **Qwen3.6-35B-A3B-origin BF16** checkpoint，目录须包含模型配置、tokenizer 和全部权重分片。当前输出格式固定为 40 层、256 个 experts、top-8、hidden size 2048。原权重约 67 GiB，裸机 4090 用 **4 × 24 GiB，TP4** 起步。显存还要容纳 KV cache、采集缓冲区和运行时开销。

下面生成一份用于 smoke 和小批量验收的配置及一条合成输入。编辑第一行的模型路径，Slurm 上则使用计算节点能访问的共享路径。

```bash
MODEL=/absolute/path/to/Qwen3.6-35B-A3B-origin
mkdir -p data/quickstart/1-8k
cat > data/quickstart/1-8k/natural.json <<'JSONL'
{"id":"quickstart-1","prompt":"Explain how a mixture-of-experts model uses its router to select experts for each token. Give one concrete example.","meta":{"domain":"technical"}}
JSONL

.venv/bin/python - "$MODEL" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path("config/collect.json").read_text())
config.update(
    model=str(Path(sys.argv[1]).expanduser().resolve()),
    source=str(Path("data/quickstart").resolve()),
    output=str(Path("data/quickstart-output").resolve()),
    tensor_parallel_size=4,
    gpu_memory_utilization=0.90,
    max_context=4096,
    max_num_batched_tokens=1024,
    max_new_tokens=32,
    per_bucket=1,
    buckets=["1-8k"],
    batch_size_by_bucket={"1-8k": 1},
    enforce_eager=True,
)
Path("data/collect.quickstart.json").write_text(json.dumps(config, indent=2) + "\n")
PY

export CONFIG="$PWD/data/collect.quickstart.json"
```

使用真实语料时，`source` 指向按 bucket 分目录的数据根目录。各 bucket 下的 `natural.json` 是 JSON Lines，每行至少包含 `id` 和 `prompt`。修改采集配置时给 `output` 用新目录，避免混用旧计划和旧结果。生产运行前先小规模验证显存和磁盘占用。

## 裸机 4090

在仓库根目录运行。将 `CUDA_VISIBLE_DEVICES` 改为四张空闲卡的编号，并保持 `tensor_parallel_size=4`。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CONFIG="$PWD/data/collect.quickstart.json"
bash scripts/smoke.sh
bash scripts/collect.sh
```

如果 TP 卡在 NCCL 初始化，先用 `nvidia-smi topo -m` 检查拓扑，用 `NCCL_DEBUG=INFO` 重跑定位。确认问题出在 GPU 间 P2P 时，可用 `NCCL_P2P_DISABLE=1 bash scripts/smoke.sh` 对照排查。[vLLM NCCL 排障文档](https://docs.vllm.ai/en/v0.32.0/usage/troubleshooting/)

## Slurm

将例子里的 `pos` 换成实际分区。仓库、虚拟环境、模型和数据需要能从计算节点访问。每个 TP 实例使用同一节点上的 GPU。

先用单个 task 跑 smoke。申请的卡数要与配置中的 `tensor_parallel_size` 一致：

```bash
export CONFIG="$PWD/data/collect.quickstart.json"
srun -p pos -N 1 -n 1 --gres=gpu:4 --cpus-per-task=8 --mem=64G \
  --time=00:30:00 bash scripts/smoke.sh
```

再提交一组 TP4 跑小批量采集。如果同一目录已跑过裸机采集，先为 Slurm 验收设置新的 `output`，以免跳过已完成样本：

```bash
sbatch -p pos --chdir="$PWD" -N 1 --ntasks=1 --gpus-per-task=4 \
  --cpus-per-task=8 --mem-per-cpu=8G --time=00:30:00 \
  --export=ALL,CONFIG="$PWD/data/collect.quickstart.json",TP_SIZE=4 \
  scripts/collect.sbatch
```

批量采集时，`--ntasks` 对应独立采集实例数，`--gpus-per-task` 对应每个实例的 TP。脚本会共用采集计划并按 task 分片。若要用 TP2，选能容纳该 checkpoint 和运行时开销的卡，并同时把配置、`TP_SIZE` 与每 task 卡数设为 2。

## 验收结果

- Smoke 成功时打印 `smoke ok`。它检查长度限制、EOG、重复终止及 prefill/decode 数值对齐。`output/_smoke/run-*/comparison.json` 保存对比结果。
- 小批量成功后，`output/1-8k/quickstart-1/` 包含专家 ID、router logits、top-k 权重、所选层 hidden states、token 对齐信息和 `stop.json`。Slurm 任务日志位于 `output/logs/<job-id>/`。
- 如果报显存或 KV cache 容量不足，先确认四张卡都空闲。可减少 `max_num_batched_tokens`、批大小或采集层数。Smoke 固定使用 4096 上下文，小批量采集使用配置里的 `max_context`。
- 升级 vLLM 时重新跑 smoke 并核对逐层数据和吞吐。适配器仍依赖 vLLM 内部接口，CUDA Graph 和 `torch.compile` 也需单独验证。
