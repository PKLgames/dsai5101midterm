# Cat Image Arena — 猫图片特征提取与偏好预测（纯 CPU）

把 1,558 张猫图编码成定长向量，并用它预测**某位观测者会选 A 还是 B**。整条流水线在无 GPU 的笔记本
（12th Gen i7-1260P，7 GB 内存）上几分钟跑完。

项目还从另一份工作（`dsai-5101-sdm-proj`，见 `README_allie_chan.md`）借用了少量可复用组件——
逐猫强度与 YOLO/CLIP 图片标签表。标签表已内置到 `data/approach_a/`，因此本仓库无需再克隆该项目；
特征族比较在同一估计器、同一套折上进行。

---

## 结果速览

### 特征流水线

| 阶段 | 实测 |
|---|---|
| 从 Hugging Face 下载 1,558 张图 | 183.8 MB，**115 s**（16 线程，断点续传） |
| 抽特征（MobileNetV3-Small @224，冻结） | **36–38 s ≈ 41–43 img/s**，峰值内存 **≈670 MB** |
| 特征质量（同猫 vs 异猫余弦） | 0.803 vs 0.692，差距 **+0.111**，AUC **0.844**，5-NN 同猫检索 **27.7%**（随机 0.38%） |

### 哪类特征有助于预测偏好

按图对分组 5 折，覆盖全部 24,896 行；统一估计器（L2 逻辑回归 + side 列，`C = 1`）。
完整指标与解读见 `docs/feature_choices.zh-CN.md`。

| 特征族 | log loss | accuracy | 两张图都未见 | 两只猫都未见 |
|---|---:|---:|---:|---:|
| 仅观测者人口学 | 0.6900 | 0.532 | 0.6901 | 0.6893 |
| 粗标签（毛色） | 0.6828 | 0.553 | 0.6823 | **0.6805** |
| 粗标签（全部） | 0.6800 | 0.560 | 0.6833 | 0.6841 |
| 手工统计（20） | 0.6865 | 0.543 | 0.6878 | 0.6892 |
| 冻结嵌入（576） | 0.6674 | 0.591 | **0.6797** | 0.6869 |
| 嵌入 + 手工 | 0.6658 | 0.592 | 0.6786 | 0.6876 |
| 逐猫记忆 | 0.6600 | 0.600 | 0.6776 | 0.6913 |
| 猫记忆 + 嵌入 + 手工（联合） | 0.6565 | 0.606 | 0.6847 | 0.6945 |
| **分流：猫+图都见过用联合，否则用毛色标签** | **0.6565** | 0.606 | **0.6823** | **0.6805** |
| 该列特征族的中位数（log loss） | — | — | 0.6837 | 0.6892 |

由此得到三条影响代码结构的结论：

1. **观测者人口学不能预测偏好**（0.6900，比常数预测器还差）。
2. **见过的猫 → 记忆更强；没见过的猫 → 外观特征更强**；因此单一联合混合模型在全新猫上反而吃亏
   （0.6945），尽管它总体最优（0.6565）。
3. **粗标签校准最好（ECE 0.007）、迁移到新猫最好**（0.6805），而稠密嵌入在见过的猫上贡献主要精度。
4. **加更多特征救不了短板。** 拼接第二个主干会把平均变好（0.6659 → 0.6607），却让新猫更差
   （0.6874 → 0.6938）；按"是否见过"分流到迁移模型能把短板补上、平均不变：
   0.6565 / 0.6823 / 0.6805，两个冷启动切片都低于特征族中位数（0.6837 / 0.6892）。

---

## 目录结构

```
dsai5101midterm/
├── README.md                     ← 英文版（本文件的英文对照）
├── README.zh-CN.md               ← 本文件（中文）
├── requirements.txt              ← 精确版本（torch 必须走 CPU 源）
├── Dataset/                      ← 课程 CSV + 说明（不含图片）
├── src/catarena/
│   ├── paths.py                  相对路径、类别水平、方案 A 位置
│   ├── index.py                  读 CSV → 图片索引 / 图对 key
│   ├── download.py               HF 并发下载、断点续传、manifest
│   ├── preprocess.py             推理用确定性 transform（224 中心裁剪）
│   ├── backbones.py              冻结主干（mnv3s / resnet18 / efficientnet_b0）
│   ├── handcrafted.py            20 维可解释低层统计
│   ├── extract.py                批处理、吞吐统计、npz + meta 缓存
│   ├── features.py               读缓存、配对设计矩阵、人口学 one-hot
│   ├── qa.py                     相似度 / PCA / 最近邻诊断
│   ├── hybrid.py                 混合模型（猫记忆 + 外观 + 标签）      [本项目原创]
│   └── approach_a/               从另一份工作复制的可复用组件
│       ├── strength.py           逐项强度 + side 项（其 fit_bt）
│       ├── labels.py             读取其 YOLO+CLIP 标签表 + 同款编码器
│       └── metrics.py            同款 log_loss / brier / accuracy
├── scripts/
│   ├── 00_setup_env.sh           纯 CPU venv（uv + download.pytorch.org/whl/cpu）
│   ├── 01_download_images.py     1,558 张图，可续传
│   ├── 02_build_index.py         image_index.csv + 重叠检查
│   ├── 03_extract_features.py    特征缓存（主干 / 尺寸 / TTA / limit 参数）
│   ├── 04_feature_qa.py          诊断与图
│   ├── 05_baseline_pairs.py      嵌入基线、两种防泄漏协议、提交文件
│   ├── 07_feature_families.py    特征族对比 + 分观测者指标
│   ├── 08_cold_start_fix.py      更强特征 vs 分流的冷启动实验
│   └── 09_hybrid_submission.py   全量拟合最终模型并写出预测
├── docs/
│   ├── pipeline_design.md        ← 设计说明（英文）
│   ├── pipeline_design.zh-CN.md  ← 同上（中文）
│   ├── feature_choices.md        ← 哪类特征有用，含指标（英文）
│   └── feature_choices.zh-CN.md  ← 同上（中文）
└── data/                         ← 运行时生成，已 gitignore（仅限教学/研究）
    ├── images/<cat>/<photo>.jpg  1,558 张，183.8 MB
    ├── manifest.csv              字节数、尺寸、来源 URL
    ├── image_index.csv           每张图的 cat / split / 出现次数
    ├── approach_a/               内置的标签表（只读）：photo_labels.csv、cat_labels.csv
    ├── features/
    │   ├── <tag>.npz             paths, deep (N,576), hand (N,20), ok
    │   ├── <tag>.meta.json       主干、版本、耗时、内存、种子
    │   ├── qa_*_<tag>.{png,json} 特征诊断
    │   ├── baseline_metrics.{json,csv}
    │   ├── feature_families.{json,md,png}
    │   ├── cold_start_fix.{json,md}
    │   └── hybrid_fit.json       最终规格、分流比例、分片统计
    └── submissions/
        ├── pred_test.csv         嵌入基线
        ├── pred_hybrid_test.csv  联合混合模型（--structure joint）
        └── pred_routed_test.csv  分流模型（推荐）

datasets/                         ← 可直接使用的表（gitignore，见 datasets/README.md）
├── train_pairs.csv.gz            24,896 × 636：特征 + 画像 + 选择
├── test_pairs.csv.gz             6,224 × 634：同列结构，无标签
└── image_features.csv.gz         1,558 × 598：单图特征
```

## 快速开始

需要 [uv](https://docs.astral.sh/uv/)。课程 CSV 放在 `Dataset/`（已包含）。图片不随项目分发，需下载。

```bash
cd dsai5101midterm

# 1) 建纯 CPU 环境（torch/torchvision 走 PyTorch CPU 源，约 200 MB）
bash scripts/00_setup_env.sh

# 2) 下载图片（183.8 MB，约两分钟）
.venv/bin/python scripts/01_download_images.py

# 3) 建索引 → 抽特征 → 诊断 → 基线
.venv/bin/python scripts/02_build_index.py
.venv/bin/python scripts/03_extract_features.py --workers 4 --threads 8
.venv/bin/python scripts/04_feature_qa.py
.venv/bin/python scripts/05_baseline_pairs.py
```

产出 `data/features/mobilenet_v3_small_224.npz`（1,558 × 576 深特征 + 1,558 × 20 手工特征）与
`data/submissions/pred_test.csv`（6,224 行）。

对比与混合模型需要的标签表已**内置**在 `data/approach_a/photo_labels.csv`，不再需要第二份仓库：

```bash
.venv/bin/python scripts/03_extract_features.py --backbone resnet18   # 可选第二主干，约 1 分钟
.venv/bin/python scripts/07_feature_families.py      # 约 3.5 分钟
.venv/bin/python scripts/08_cold_start_fix.py        # 约 7 分钟
.venv/bin/python scripts/09_hybrid_submission.py     # 约 30 秒，写分流版提交
```

另外两个入口：

```bash
# 单张图 → 特征，并核对与缓存是否一致
.venv/bin/python scripts/10_quickstart.py --image data/images/cat_000002/cat_000002_0001.jpg

# 直接生成「特征 + 观测者画像 + 选择」的训练/测试表
.venv/bin/python scripts/06_export_datasets.py
```

## 脚本

| 脚本 | 关键参数 | 作用 |
|---|---|---|
| `00_setup_env.sh` | `PYTHON_TAG`, `TORCH_VER` | 建 `.venv`；CPU 轮子避免约 2.5 GB 的 nvidia 依赖 |
| `01_download_images.py` | `--workers`, `--limit` | 16 线程、3 次退避重试、`.part` 原子落盘、已存在则跳过 |
| `02_build_index.py` | — | 生成 `image_index.csv`，检查 train/test 重叠与缺图 |
| `03_extract_features.py` | `--backbone`, `--size`, `--batch-size`, `--workers`, `--threads`, `--tta-flip`, `--limit`, `--tag`, `--write-csv`, `--list-backbones` | 抽特征、写缓存、报告吞吐 |
| `04_feature_qa.py` | `--tag`, `--no-plots` | 同猫/异猫相似度、5-NN 检索、PCA、最近邻图 |
| `05_baseline_pairs.py` | `--tag`, `--C`, `--p2-folds`, `--seed` | 图对分组 CV + 图片留出，写提交文件 |
| `06_export_datasets.py` | `--tag`, `--out-dir`, `--no-images` | 生成 `datasets/{train,test}_pairs.csv.gz` 与 `image_features.csv.gz` |
| `07_feature_families.py` | `--folds`, `--p2-folds`, `--seed`, `--no-coldstart`, `--replot` | 十类特征族、三种协议、分观测者与校准指标 |
| `08_cold_start_fix.py` | `--reuse` | 多主干特征与分流结构对着冷启动短板的实验 |
| `09_hybrid_submission.py` | `--structure routed\|joint`, `--spec`, `--C`, `--sweep` | 全量拟合最终模型并写出预测 |
| `10_quickstart.py` | `--image`, `--batch`, `--backbone`, `--tag` | 最小的「图片→特征」示例，并核对缓存 |

可选主干（接口一致，一条参数切换）：

| 主干 | 维度 | 参数 | 吞吐（128 张探针） | 峰值内存 |
|---|---|---:|---:|---:|
| `mobilenet_v3_small` | 576 | 2.5 M | **24.2 img/s** | 654 MB |
| `resnet18` | 512 | 11.7 M | 15.2 img/s | 766 MB |
| `efficientnet_b0` | 1280 | 5.3 M | 17.3 img/s | 1.08 GB |

## 特征缓存格式

`data/features/<tag>.npz`（`np.load` 直读，无 pickle）：

| 键 | 形状 | 含义 |
|---|---|---|
| `paths` | (N,) str | 相对 `data/images/` 的路径，与 `a_image` / `b_image` 逐字符相同 |
| `deep` | (N, 576) f32 | L2 归一化的冻结 CNN 池化特征 |
| `hand` | (N, 20) f32 | 可解释低层统计（meta 里的 `hand_features`） |
| `ok` | (N,) bool | 是否成功解码（本数据集 1,558/1,558） |
| `hand_names` | (20,) str | 列名 |

`<tag>.meta.json` 记录主干与权重文件、输入尺寸、归一化常数、torch/torchvision 版本、线程数、
批大小、耗时、img/s、峰值内存、CPU 型号、种子与时间戳——报告里「硬件与运行时间」一栏可直接引用。

维度：**单图 596**（576 深度 + 20 手工）；图对行 **578**（`f_A − f_B` + `comparison_type`），
带 26 维人口学 one-hot 时 **604**，用 `diff_absdiff` 时 **1180**。

## 可直接使用的表（`datasets/`）

`scripts/06_export_datasets.py` 把**图片特征、观测者画像、观测到的选择**拼成一行一条响应，
不写代码也能直接训练：

| 文件 | 规模 | 内容 |
|---|---|---|
| `train_pairs.csv.gz` | 24,896 × 636 | `d_0000…d_0575`（嵌入 A−B）+ `h_*`（20）+ 6 列原始画像 + 26 列画像 one-hot + `choice`、`y` |
| `test_pairs.csv.gz` | 6,224 × 634 | 同列结构，无标签 |
| `image_features.csv.gz` | 1,558 × 598 | 单图特征表，用于还原单侧向量 |

已验证：直接用 `train_pairs.csv.gz` 训练 `LogisticRegression(C=0.003)`，可复现流水线的 **0.665** log loss。
完整列说明见 [datasets/README.zh-CN.md](datasets/README.zh-CN.md)。

## 使用特征

```python
import sys; sys.path.insert(0, "src")
from catarena import features, index

feats = features.load_features("mobilenet_v3_small_224")   # None = 取最新缓存
X, present_a, present_b = features.pair_matrix(df, feats, mode="diff", with_demo=True)
y = features.label_a(df)        # 选 A 记为 1
# X: (n_rows, 604) = 576 维嵌入差值 + 26 维人口学 + 2 维 comparison_type
```

图片与特征靠**路径字符串**对应，不靠行号；`lookup()` 对缺失图片返回零向量并给出 `present=False`
（train/test 共 31,120 个图片位置实测缺失 0 个）。

图对设计矩阵在内存中现算（约 0.1 s），不落盘，所以换编码方式无需重导任何文件。

## 混合模型

`catarena.hybrid.HybridModel` 是一次 L2 逻辑回归，输入为差值编码的若干块：

```
[ 猫 one-hot（仅拟合中出现过的猫） | 576 维嵌入 | 20 维手工（z-score） | 标签 | side ]
```

* 猫记忆来自 `catarena.approach_a.strength`；未见过的猫贡献 0。
* 外观块由本项目生成，承担新图与同猫内部（猫差值恒为 0）的信号。
* 标签块可选，读取另一份工作的 `prepared/photo_labels.csv`。
* `scripts/09` 用 `猫 + 嵌入 + 手工`、`C = 0.3` 在全部 24,896 行（834 列）上拟合，
  写出 `data/submissions/pred_hybrid_test.csv`。

实测权衡（详见 `docs/feature_choices.zh-CN.md` §6b）：猫见过时联合模型最好（总体 0.6565），
但对**全新猫**它比纯嵌入更差（0.6945 vs 0.6869），因为猫哑变量吃掉了猫级方差。加第二个主干救不了
（新猫 0.6938）；**分流可以**：

* 猫和两张图都出现在训练集时用联合模型（测试集 50% 的行）；
* 其余行用一个只用毛色 shares 的迁移模型。

结果：平均不变（0.6565），两图未见 0.6823，两猫未见 0.6805——都低于特征族中位数
0.6837 / 0.6892。`scripts/09_hybrid_submission.py --structure routed` 会写出
`data/submissions/pred_routed_test.csv`。

## 借用组件的来源

放在独立子包，边界一眼可见；只复制可复用部分，不包含任何重跑对方分析的代码。

| 文件 | 来源 |
|---|---|
| `src/catarena/approach_a/strength.py` | 改写自其 `analysis/03_cat_strength_model.ipynb`（`fit_bt`、`predict_bt`） |
| `src/catarena/approach_a/labels.py` | 读取其 `photo-labelling/*.py` 产出的 `prepared/photo_labels.csv`；编码器对齐其 `numeric_diff` / `one_hot_diff` / `with_side` |
| `src/catarena/approach_a/metrics.py` | 其 `log_loss` / `brier` / `accuracy` 定义 |

除此之外不导入该项目任何内容，本项目也从不写入对方目录。`catarena/approach_a/` 之外全部为本项目原创。

## 可复现性

* 固定：路径字典序、`torch.manual_seed(0)`、折的种子（基线与特征族对比均为 `0`）、
  报告处的 `C` 由图对分组留出集选出。
* 版本锁定在 `requirements.txt`；每次抽特征把实际环境写进 `<tag>.meta.json`。
* 已知数值波动：torch 线程数改变浮点求和顺序（log loss 第 4 位小数）；其余可逐位复现
  （无随机增强、无 dropout、无 GPU 非确定性）。重跑 `05` 得到完全相同的 `baseline_metrics.json`；
  重跑 `07` 得到相同的 `feature_families.json`（KFold 有种子、估计器确定）。
* 若沙箱禁止 POSIX 信号量，DataLoader 多进程不可用，抽取脚本会自动回退 `num_workers=0`
  （本机 43 img/s），并在 meta 里记录 `num_workers_requested` 与 `num_workers`。

## 局限

* 人类偏好噪声很大：两位观测者在重复图对上的答案只有 55.7% 一致。任何模型的上限都远低于"确定"，
  在分歧图对上所有特征族的准确率都在 0.50。
* 没有标注者 ID，只有 70 种人口学画像，因此"不同观测者"只是近似——这也解释了为什么人口学特征无用。
* 两种冷启动（新图 vs 新猫）给出相反排名，报告时必须两种都写。
* 测试集标签未公开，上述数字都不是排行榜分数。
* 低层手工特征几乎不提升精度，价值在于可解释性。

## 发布到 GitHub

该目录已经是带有一次提交的 git 仓库，尚未推送到任何远端。

```bash
cd dsai5101midterm
git status                      # 应为干净状态

# 1) 先在 GitHub 上建一个空仓库（不要勾选 README / .gitignore），然后
git remote add origin git@github.com:<你的用户名>/<仓库名>.git
#   或用 HTTPS：
#   git remote add origin https://github.com/<你的用户名>/<仓库名>.git

git branch -M main
git push -u origin main
```

装了 GitHub CLI 的话，前两步可合并为
`gh repo create <仓库名> --private --source=. --push`。

**没有**进仓库的内容（见 `.gitignore`）：183 MB 图片、特征缓存、78 MB 的 `datasets/*.csv.gz`、
`.venv`、以及课程名单。它们都能重新生成：
`git clone` → `bash scripts/00_setup_env.sh` → `scripts/01` → `scripts/03` → `scripts/06`。

如果需要分发大表，三选一：

| 方式 | 做法 |
|---|---|
| GitHub Release 附件 | 建一个 release，上传 `datasets/*.csv.gz`（单文件均 <100 MB） |
| Git LFS | `git lfs track "datasets/*.csv.gz" && git add .gitattributes && git add datasets/*.csv.gz && git commit` |
| 课程提交包 | 把 `datasets/` 与代码一起打包提交 |

提交作者在本地被设为 `pklgames <pklgames@users.noreply.github.com>`；推送前可用
`git config user.name "..."`、`git config user.email "..."` 改成你自己的名字。
