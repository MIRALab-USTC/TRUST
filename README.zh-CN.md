<h1 align="center">🚗 TRUST：面向自动驾驶的鲁棒离线强化学习<br>缓解分布外动力学下的重尾问题</h1>

<p align="center"><em>面向自动驾驶的轨迹约束鲁棒离线强化学习。</em></p>

<p align="center">
  <img src="assets/trust_architecture.png" width="100%" alt="TRUST 架构：鲁棒目标近似、基于 IQR 的损失重加权和 Actor–Critic 学习">
</p>

<p align="center">
  <a href="https://github.com/MIRALab-USTC/TRUST"><img src="https://img.shields.io/badge/Code-TRUST-111827?style=for-the-badge&amp;logo=github" alt="代码仓库：TRUST"></a>
  <img src="https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?style=for-the-badge&amp;logo=python&amp;logoColor=white" alt="Python 3.11 或 3.12">
</p>

<p align="center"><a href="README.md">English</a></p>

---

## 📚 项目概览

**TRUST** 通过两项关键机制应对自动驾驶中的分布外动力学：**轨迹约束鲁棒目标**和**基于 IQR 的 critic 重加权**。方法近似计算车辆轨迹特征的局部最坏情况变化，并提高上尾 critic 损失对应样本的权重。在离线 Actor–Critic 框架中，策略使用优势加权回归进行更新。该 JAX/Flax 实现支持从已保存的状态转移数据进行离线训练，以及在 ScenarioMax 处理后的场景中进行闭环评估。

该方法对应论文 *Robust Offline RL for Autonomous Driving: Mitigating Heavy Tails under OOD Dynamics*。

## 安装

需要 Python 3.11 或 3.12。执行以下命令，安装 TRUST 及闭环评估所需的依赖：

```bash
git clone https://github.com/MIRALab-USTC/TRUST.git
cd TRUST
python -m venv .venv
source .venv/bin/activate
pip install -e ".[evaluation]"
```

使用 GPU 训练时，请安装与硬件匹配的 JAX 加速运行环境。

## 训练

提供已保存的 NPZ 数据集或包含 NPZ 分片的目录。数据必须包含 `observations`、`actions`、`rewards`、`next_observations`、`terminals`、`timeouts` 和 `trajectory_mask`。动作需归一化至 `[-1, 1]`，轨迹掩码需选中后继观测中有效的车辆轨迹特征。导出数据时应使用与策略一致的观测配置。

```bash
python -m vmax.scripts.training.train_offline \
  dataset=womd \
  offline_dataset_path=/path/to/offline_train.npz \
  seed=0 \
  name_run=TRUST_WOMD_seed0
```

使用 `dataset=nuplan` 或 `dataset=nuscenes` 可切换对应的数据集配置。默认配置要求 2,000,000 条状态转移样本；如果数据量不同，可设置 `algorithm.expected_dataset_size=N`，或通过 `algorithm.expected_dataset_size=null` 关闭数量检查。模型检查点和配置保存在 `runs/<name_run>/` 中。

## 闭环评估

使用 ScenarioMax 处理后的 TFRecord 场景评估训练好的模型：

```bash
python -m vmax.scripts.evaluate.evaluate \
  --path_model runs/TRUST_WOMD_seed0 \
  --path_dataset /path/to/scenariomax/validation.tfrecord \
  --batch_size 8
```

TFRecord 中需要包含自车路径。默认形状为 10 条路径、每条 300 个点；如果实际形状不同，可通过 `--num_paths` 和 `--num_points_per_path` 指定。默认评估时长为 80 步。报告保存在 `benchmark/<run>/<dataset-identity>/<checkpoint>/` 中，也可以通过 `--output_dir` 指定目录。
