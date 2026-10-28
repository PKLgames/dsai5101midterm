# 可直接使用的训练 / 测试表

由 `scripts/06_export_tables.py` 从特征缓存与课程 CSV 生成。一行一条标注者响应，
不运行本仓库任何代码也能直接训练模型。

| 文件 | 行 × 列 | 体积 | 内容 |
|---|---:|---:|---|
| `train_pairs.csv.gz` | 24,896 × 636 | 58 MB | 图片特征 + 观测者画像 + 观测到的 `choice` |
| `test_pairs.csv.gz` | 6,224 × 634 | 15 MB | 同列结构，去掉标签 |
| `image_features.csv.gz` | 1,558 × 598 | 3.4 MB | 单图特征表（576 维深度 + 20 维手工） |

重新生成：

```bash
make tables                                             # 默认 MobileNetV3-Small 缓存
python3 scripts/06_export_tables.py            # 同上（已激活 venv）
python3 scripts/06_export_tables.py --tag resnet18_224 --out-dir tables/resnet18
```

## 列说明

| 块 | 列 | 含义 |
|---|---|---|
| 标识 | `train_row_id` / `test_row_id`、`a_cat`、`b_cat`、`a_image`、`b_image`、`comparison_type` | 该条响应与两张图（`a_image` 是相对 `artifacts/images/` 的路径） |
| 观测者画像 | `gender`、`pet_history`、`pet_care`、`pet_contact`、`future_pet`、`mbti_ie` | 原始取值，空值写为 `missing` |
| 画像 one-hot | `demo_<字段>_<水平>`（共 26 列） | 模型实际使用的编码，含 `missing` 水平 |
| 图片特征 | `d_0000 … d_0575` | 嵌入差值 `f(A) − f(B)`，每张图已做 L2 归一化 |
| 低层特征 | `h_<名称>`（20 列） | 手工统计量的差值，同样 A−B |
| 标签（仅训练集） | `choice`、`y` | `"A"`/`"B"` 与 1/0 |

`test_pairs.csv.gz` 不含 `choice` / `y`。

## 直接使用

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("tables/train_pairs.csv.gz")
feat = [c for c in train.columns if c.startswith(("d_", "demo_"))]
clf = LogisticRegression(C=0.003, max_iter=400).fit(train[feat], train["y"])
```

实测：这两行代码在训练表上做图对分组 5 折可得 **0.665 log loss**，与仓库内流水线的结果一致，
说明导出的列就是模型实际使用的列。

## 注意

* 特征是**差值** `f(A) − f(B)`，不是单侧向量；需要单侧特征时用 `a_image`/`b_image`
  与 `image_features.csv.gz` 的 `image_path` 做连接。
* 全部为 float32，按 6 位有效数字输出。
* 测试表没有标签（课程方未公开），因此无法据此评估提交分数。
* 标签与画像列来自 `course/train.csv`、`course/test.csv`；特征来自
  `artifacts/features/<tag>.npz`（见 `scripts/03_extract_features.py`）。
* 这些文件**未提交进 git**（见 `.gitignore`）：58 MB 会让仓库臃肿。请作为 release 附件、
  用 Git LFS，或放进课程提交包。
