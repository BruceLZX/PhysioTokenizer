# PhysioTokenizer — HuggingFace 运行指南

## 前置准备

### 1. 创建 HuggingFace 账号
访问 https://huggingface.co/join 注册

### 2. 获取 Access Token
https://huggingface.co/settings/tokens → New Token → Write 权限

### 3. 本地安装 HF CLI
```bash
pip install huggingface_hub
huggingface-cli login
# 粘贴你的 token
```

## 方法一：HF Spaces（推荐，有 Web UI）

### 创建 Space
1. 访问 https://huggingface.co/new-space
2. Space name: `physiotokenizer`
3. SDK: **Docker**
4. Docker template: **Blank**
5. Space hardware: **A10G small** ($1.05/hr) 或 **T4 medium** (免费/$0.60/hr)
6. Create Space

### 上传代码
```bash
# Clone HF Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/physiotokenizer
cd physiotokenizer

# Copy all project files
cp -r ~/CodingProject/PhysioTokenizer/* .
cp -r ~/CodingProject/PhysioTokenizer/paper .

# Push
git add .
git commit -m "Initial PhysioTokenizer project"
git push
```

### 在 HF Space 中运行
Space 会自动 build Docker image 并运行。你也可以 SSH 进去：

```bash
# 在 Space Settings → Factory Rebuild 之后
# 等待 build 完成（约 5-10 分钟）

# 然后通过 SSH 或 Space 的 terminal 运行：
python scripts/run_on_hf.py --mode all
```

## 方法二：HF GPU 实例（更灵活）

### 创建 GPU 实例
访问 https://huggingface.co/settings/billing → 选 GPU 实例

### 运行步骤
```bash
# 1. SSH 进 GPU 实例
ssh hf-user@your-instance

# 2. Clone 代码
git clone https://github.com/BruceLZX/PhysioTokenizer.git
cd PhysioTokenizer

# 3. 安装依赖
pip install -r requirements.txt

# 4. 下载数据（PTB-XL 走 PhysioNet 自动下载）
python -c "
import wfdb
wfdb.dl_database('ptb-xl', dl_dir='data/ptbxl')
"

# 5. 跑训练
python scripts/run_on_hf.py --mode train --config ecg_single_band

# 6. 跑 benchmark
python scripts/run_on_hf.py --mode benchmark \
    --model checkpoints/physiotokenizer_best.pt

# 7. 生成论文图
python scripts/run_on_hf.py --mode figures \
    --results results/benchmark_results.json

# 8. 或一键全跑
python scripts/run_on_hf.py --mode all
```

## 分阶段运行

### Phase 1: ECG 单模态（~2-4 小时，A10G）
```bash
# 最小实验：只跑 ECG
python scripts/run_on_hf.py --mode all --config ecg_single_band
```
- 数据: PTB-XL (21K records)
- 模型: ~15M params
- 预期训练时间: 2-4h on A10G
- 产出: checkpoints/physiotokenizer_best.pt + results/ + figures/

### Phase 2: 加自适应边界（~4-6 小时，A10G）
```bash
# 修 configs/ecg_single_band.yaml: use_adaptive_boundaries: true
python scripts/run_on_hf.py --mode train --config ecg_single_band
```

### Phase 3: 多模态全量（~8-12 小时，A100-40G）
```bash
python scripts/run_on_hf.py --mode all --config full_multimodal
```
- 数据: ECG + EEG + PPG
- 模型: ~50M params
- 需要 A100 (40GB VRAM)

## 产出文件

运行完成后，你会得到：

```
checkpoints/
  physiotokenizer_best.pt     ← 训练好的模型权重

results/
  benchmark_results.json      ← 所有指标（MSE, Acc, F1, Compression...）
  benchmark_table.tex         ← AAAI 格式的 LaTeX 结果表

figures/
  fig1_reconstruction.pdf     ← 重建质量对比
  fig2_downstream.pdf         ← 下游任务性能
  fig3_tokens.pdf             ← Token 嵌入空间可视化
  fig4_ablation.pdf           ← 消融实验
  fig5_cross_modal.pdf        ← 跨模态检索

paper/
  physiotokenizer.tex         ← AAAI 2027 格式论文（插入结果后）
```

## 下载结果到本地

```bash
# 从 HF Space 下载
huggingface-cli download YOUR_USERNAME/physiotokenizer \
    --include "results/*" "figures/*" "checkpoints/*" \
    --local-dir ./results/

# 或从 GitHub（如果上传了）
git pull
```
