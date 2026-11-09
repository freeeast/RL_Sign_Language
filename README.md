# SignBot 在 AutoDL 上复现（review 与修复版）

本仓库不直接拷贝 [qiaoguanren/Signbot](https://github.com/qiaoguanren/Signbot)（约 700MB），而是：

- 固定上游 commit `55087d2`
- 在上面打 4 个补丁（`patches/`），修掉导致跑不起来的问题
- 提供 AutoDL 一键脚本（`autodl/`），覆盖环境搭建、How2Sign 数据处理到训练的全流程

```
patches/                  对上游的修改（git format-patch，可直接 review diff）
autodl/env.sh             所有路径集中在这里（默认 /root/autodl-tmp/...）
autodl/01_get_signbot.sh  拉取上游代码 + 打补丁 + 链接 SMPL-X 模型
autodl/02_setup_signbot_env.sh   conda 环境 signbot：py3.8 + torch2.3.1(cu118) + IsaacGym
autodl/03_setup_dexret_env.sh    conda 环境 dexret：手部重定向 dex-retargeting
autodl/04_data_pipeline.sh       How2Sign SMPL-X → H1 身体 + Linker 手 → keybody
autodl/05_train.sh               冒烟测试 / 正式训练
autodl/check_env.py              环境自检
tests/                    数据处理部分的离线单元测试
```

---

## 1. Review 结论：上游代码在你的机器上会卡在哪里

| # | 严重程度 | 位置 | 问题 | 修复 |
|---|---|---|---|---|
| 1 | **阻塞** | `h1_mimic_view_motion.py` | README 第 6 步生成 keybody 的环境强制 `no_keybody=False`，但它要生成的 `*_key_bodies.npy` 本来就还不存在，于是 MotionLib 直接 `exit()`。另外保存 keybody 的那行代码**被注释掉了**，实际只会保存 W1 机器人的数据，所以训练要用的文件永远生成不出来 | 改成 `no_keybody=True`，恢复保存逻辑（保存在每个 motion npy 旁边）。新增 `gen_keybody.py`，用零动作驱动回放环境，不再需要"先训 1 轮再杀掉"拿一个 dummy 模型 |
| 2 | **阻塞** | `data_process_body.py` | 依赖 `ASE/ase/poselib/data/npy/S000003_P0000_T00.npy`（作者自己的 CSL-Daily 处理结果，用来固定下半身姿态），这个文件**不在仓库里** | 默认用规范模板代替（骨盆、腿、Spine3 都取单位旋转，也就是直立、面朝前方、腿伸直）。如果你有某个处理好的 npy 想当模板，可以用 `--template` 指定 |
| 3 | **阻塞** | `motion_lib.py`、`data_process_*.py`、`generate_yaml.py`、`retarget_smpl_to_h1.json`、各 yaml | 到处硬编码 `/data1/guanren/...` 和 `/home/guanren/...`；手部数据路径还写死成了 CSL-Daily | 全部改成命令行参数。yaml 里 `root` 写绝对路径，另加顶层 `hand_root`，MotionLib 从这里读手部数据 |
| 4 | **静默错误** | `data_process_body.py` / `data_process_hand.py` | How2Sign 的帧文件名是 `<clip>_<i>_3D.pkl`（没有补零），上游用 `sorted(os.listdir())` 排序，得到的顺序是 0,1,10,100,101,…,11，**动作帧被打乱了**，但程序不会报错 | 按帧号数值排序；帧号不连续时打印警告 |
| 5 | **静默错误** | `motion_lib.py` | 30 维手部关节角之间做插值时用的是四元数 slerp。由于点积 \|dot\|≥1，基本退化为"直接取前一帧"，还有可能翻转符号 | 改为线性插值 |
| 6 | 安全 | `train.py`、`helpers.py` | 写死了**作者的 WANDB_API_KEY** 和 entity。不改的话，你的训练日志会传到别人的账号下 | 删除。默认用 wandb offline 模式；执行 `wandb login` 后自动切到 online。这个 key 是由 `01_get_signbot.sh` 用 sed 删掉的，本仓库里不包含它 |
| 7 | 崩溃 | `task_registry.py` | `--resume` 时 `reset_std(..., 21)` 写死了 21 维动作，但 Linker 手策略是 55 维，续训必定报错 | 改为按实际动作维度 |
| 8 | 崩溃/体验 | `motion_lib.py` | 第一次运行生成 pkl 缓存后直接 `exit()`，必须再跑一次；`data/pkl` 目录不存在时直接报错（README 里写的 `mkdir pkl` 位置也不对）；yaml 改了之后仍然读旧 pkl，导致手部和身体数据错位 | 自动建目录；生成缓存后继续运行；检测到缓存过期时自动重建；新增断言：手部帧数必须等于身体帧数 |
| 9 | 性能 | `retarget_motion_h1_*.py` | `multiprocessing.cpu_count()` 在容器里返回的是**宿主机**核数（可能上百），会起几十上百个同时 import torch+isaacgym 的进程，导致内存爆掉 | 默认 `min(8, 实际可用核数)`，可用 `--workers` 调整 |
| 10 | 性能 | `data_process_*.py` | 每个 clip 都重新创建一次 SMPL-X 模型，而且逐帧前向 | 模型只创建一次，在 GPU 上按 clip 批量前向；身体和手部共用同一次前向结果（保证两者帧数一致） |
| 11 | 性能 | `h1_mimic_hand.py::_set_env_state` | 每次 reset 都用 Python 循环逐个 env 赋值（4096 个 env 就是数千次 CUDA 小 kernel） | 向量化，语义不变（有单测对比） |
| 12 | 小问题 | `play.py`、`logger.py`、`base_task.py`、`requirements.txt` | `num_envs` 写死成 884；日志路径依赖当前目录；headless 模式下录视频没有图形设备；`requirements.txt` 是 200 行的 `pip freeze`（包含 modelscope、streamlit、ray…，并且是 torch2.4+cu121） | 修正；另给出最小依赖 `autodl/requirements_signbot.txt`，并用 `opencv-python-headless`，服务器上不需要 libGL |

另外两点需要注意：

- **README 写的重定向脚本不对**。手语数据应该用 `retarget_motion_h1_smpl.py`（SMPL→H1），而 README 写的 `retarget_motion_h1_all.py` 是 CMU 动捕用的。
- **上游没有放预训练模型**。README 里的 `060-40` 在仓库中不存在，只能自己训练。

数据处理的数学部分（坐标翻转、冻结根节点和下半身、速度计算、手部对齐、向量重定向）**与上游完全一致**。`tests/test_pipeline.py` 会把新实现和从上游代码里抄出来的参考实现逐项比对。

---

## 2. AutoDL 部署步骤

### 0) 前提（你已经完成）

```
/root/autodl-tmp/IsaacGym_Preview_4_Package.tar
/root/autodl-tmp/SMPLX_NEUTRAL.npz   (以及 .pkl，不是必需的)
/root/autodl-tmp/data/raw_poses/     How2Sign val 的逐帧 SMPL-X pkl（下面有几层子目录都可以，脚本会递归查找）
```

如果 `raw_poses` 里还是压缩包，需要先解压。路径和上面不一样的话，可以改 `autodl/env.sh`，或者用环境变量覆盖，例如 `RAW_POSES=/xxx bash autodl/04_data_pipeline.sh`。

### 1) 拉取本仓库和打好补丁的 Signbot

```bash
source /etc/network_turbo
```

```bash
cd /root/autodl-tmp && git clone -b claude/cool-cannon-90aot1 https://github.com/freeeast/RL_Sign_Language.git
```

```bash
cd /root/autodl-tmp/RL_Sign_Language && bash autodl/01_get_signbot.sh
```

完成后得到 `/root/autodl-tmp/Signbot`，它是 `55087d2` 加上 5 个本地 commit。可以用 `git log` / `git diff 55087d2` 查看所有改动。

### 2) 环境（约 10–20 分钟，主要时间花在下载 torch）

```bash
bash autodl/02_setup_signbot_env.sh
```

```bash
bash autodl/03_setup_dexret_env.sh
```

`02` 结束时会运行 `check_env.py`，检查以下各项：IsaacGym 导入、gymtorch 编译、GPU PhysX 建 sim、numpy==1.23.5、SMPL-X 模型、pose 数据和 pkl 键名。**全部显示 `[ok]` 再继续。**

为什么选 torch 2.3.1+cu118：版本和上游 README 一致；CUDA 版本对上镜像自带的 11.8 工具链（gymtorch 在本地做 JIT 编译）。

### 3) 数据处理

建议先用 50 个 clip 把整条流程跑通：

```bash
LIMIT=50 bash autodl/04_data_pipeline.sh
```

没有问题再处理全部（已完成的 clip 会自动跳过）：

```bash
bash autodl/04_data_pipeline.sh
```

各步骤可以单独运行：`prepare | yaml | body | hands | finalize | keybody`。

| 步骤 | 环境 | 输入 → 输出（都在 `/root/autodl-tmp/data/how2sign/` 下） |
|---|---|---|
| prepare | signbot | `raw_poses` → `source_body/*.npy`、`hand_joints/*.npz`、`index.csv` |
| yaml | signbot | → `Signbot/ASE/ase/poselib/data/configs/motions_how2sign_val.yaml` |
| body | signbot | `source_body` → `target_body/*.npy`（H1） |
| hands | dexret | `hand_joints` → `target_linker_hand/*_hand_full_pose.npy` |
| finalize | signbot | 只保留身体和手都处理成功的 clip，重写 yaml，另生成一个 16 条的 debug yaml |
| keybody | signbot + GPU | IsaacGym 回放 → `target_body/*_key_bodies.npy` |

### 4) 冒烟测试（64 个 env、20 次迭代、不连 wandb）

```bash
bash autodl/05_train.sh smoke
```

第一次运行会先生成 motion 的 pkl 缓存（`ASE/ase/poselib/data/pkl/`），然后直接开始训练。

### 5) 正式训练

建议在 tmux 里跑，这样 SSH 断开也不会中断训练：

```bash
tmux new -s train
```

```bash
cd /root/autodl-tmp/RL_Sign_Language && bash autodl/05_train.sh 000-01-how2sign
```

- 模型保存在 `/root/autodl-tmp/Signbot/legged_gym/logs/h1/000-01-how2sign/model_*.pt`
- 默认 4096 个 env，`max_iterations=25000`。显存不够时用 `NUM_ENVS=2048`，想先跑短一点就加 `--max_iterations 5000`
- 续训：`bash autodl/05_train.sh 000-02-how2sign --resume --resumeid 000-01`
- wandb：默认 offline，日志在 `Signbot/legged_gym/logs/wandb`。想用 online，先执行 `wandb login`，再加 `--entity <你的账号>`

### 6) 查看结果（服务器没有显示器）

- `play.py` 默认会开 viewer，在 AutoDL 上不能用。可以用 `--web` 配合端口转发看 web viewer，或者用 `--headless --record_video` 录 mp4（需要容器支持 Vulkan 离屏渲染，不一定可用）。
- 也可以把 `model_*.pt` 下载到有显示器的机器上，用同一套代码执行 `python play.py 000-01 --task h1_mimic_hand`。

---

## 3. 常见问题

| 现象 | 处理 |
|---|---|
| `ImportError: libpython3.8.so.1.0` | `conda deactivate && conda activate signbot`。02 脚本已经在 activate.d 里设置了 `LD_LIBRARY_PATH`，重新激活后才会生效 |
| `module 'numpy' has no attribute 'float'` | numpy 被别的包升级了：`pip install numpy==1.23.5` |
| GitHub 或 download.pytorch.org 很慢、超时 | 先执行 `source /etc/network_turbo`（AutoDL 学术加速），脚本里也已经自动开启 |
| `MotionLib ... hand has X frames but body has Y` | 这个 clip 的手部和身体不是用同一批帧处理出来的。删掉该 clip 在 `target_linker_hand` 里的旧文件，再重跑 `hands` |
| `No key body positions file found` | 先运行 `bash autodl/04_data_pipeline.sh keybody` |
| `prepare` 报 `KeyError: smplx_body_pose` | 你的 pose 文件键名和 SOKE / Neural Sign Actors 的格式不一样。把 `check_env.py` 打印出来的 keys 发给我 |
| CUDA OOM | `NUM_ENVS=2048` 或 `1024` |
| 系统盘（30G）快满了 | conda 和 pip 的缓存已经放到 `/root/autodl-tmp/.cache` 下；可以执行 `conda clean -a` |

---

## 4. 仍需在 GPU 上验证的事项（如实说明）

我这边没有 GPU，也没有 IsaacGym，所以：

- **已验证**：
  - 补丁能干净地打到上游 `55087d2` 上，并且结果与我 review 的代码树完全一致
  - 数据处理部分的单元测试全部通过：帧排序、与上游实现逐项一致的运动学计算、手部对齐、yaml/MotionLib 的路径逻辑、向量化后的 `_set_env_state` 与原循环结果一致
  - 所有 Python 文件都能编译通过，所有 shell 脚本的语法都正确
- **未验证**：IsaacGym 仿真、`gen_keybody.py`、训练和 play 的实际运行，以及 dex-retargeting 的真实输出。请按上面的顺序执行，哪一步报错就把完整日志发给我。
- **和论文设置可能存在的差异**：
  1. 下半身用的是规范模板，而不是作者那个找不到的 `S000003`
  2. 手部插值改成了线性插值
  3. 上游把 How2Sign 统一标成 30fps，本流程保持这一点不变（SOKE 会把帧率高于 24fps 的 clip 降采样到 24fps）

## 5. 本地跑单元测试（可选，不需要 GPU）

```bash
pip install numpy scipy pyyaml torch dill
```

```bash
python tests/test_pipeline.py /root/autodl-tmp/Signbot && python tests/test_yaml_motionlib.py /root/autodl-tmp/Signbot
```
