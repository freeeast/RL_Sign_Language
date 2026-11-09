# SignBot 代码结构与训练算法说明

基于上游 `qiaoguanren/Signbot@55087d2` + 本仓库 `patches/`。路径都相对于打完补丁的 `Signbot/`。

## 0. 一张图看懂数据流

```
How2Sign 3D 姿态 (Neural Sign Actors, 每帧一个 SMPL-X pkl)
   │  tools/prepare_smplx_poses.py           [signbot 环境, GPU 上跑 SMPL-X 前向]
   ├─► source_body/<clip>.npy     14 关节 SMPL 骨架动作 (SkeletonMotion)
   └─► hand_joints/<clip>.npz     左右手各 21 个关节点 (相对手腕)
   │
   │  tools/generate_motion_yaml.py → ASE/ase/poselib/data/configs/motions_how2sign_val.yaml
   │
   ├─ ASE/ase/poselib/retarget_motion_h1_smpl.py   [signbot 环境]  身体: SMPL → Unitree H1
   │     └─► target_body/<clip>.npy
   ├─ tools/retarget_hands.py                       [dexret 环境]   手: 人手 → Linker 灵巧手
   │     └─► target_linker_hand/<clip>_hand_full_pose.npy   (T, 30) = 左 15 + 右 15 个关节角
   │
   │  legged_gym/legged_gym/scripts/gen_keybody.py  [IsaacGym 运动学回放]
   │     └─► target_body/<clip>_key_bodies.npy      12 个关键 body 相对根节点的位置
   ▼
legged_gym/legged_gym/scripts/train.py --task h1_mimic_hand   (PPO, rsl_rl)
   └─► legged_gym/logs/h1/<exptid>/model_<it>.pt
```

## 1. 目录与文件作用

### 1.1 顶层

| 路径 | 作用 |
|---|---|
| `tools/`（**本仓库新增**） | 数据处理脚本，所有路径都通过参数传入，用来替代上游写死路径的 `data_process_body.py`、`generate_yaml.py`、`dex-retargeting/example/data_process_hand.py` |
| `data_process_body.py`、`generate_yaml.py` | 上游原版，路径写死为 `/data1/guanren`，**不再使用**，只作对照 |
| `api.py`、`request.py` | 论文中的"Responder"部分：用 Flask 包装本地 LLM（GLM-4-9B）做问答，和 RL 训练无关 |
| `body_models/` | `smplx/SMPLX_NEUTRAL.npz`（由 01 脚本做软链接）；`hands/linker_hand/` 和 `hands/inspire_hand/` 下放灵巧手的 URDF 和 mesh，供手部重定向使用 |
| `requirements.txt` | 作者环境的 `pip freeze`（200 行），**不使用**，改用 `autodl/requirements_signbot.txt` |
| `img/` | README 用的图片 |

### 1.2 `tools/`（新增的数据流水线）

| 文件 | 作用 |
|---|---|
| `prepare_smplx_poses.py` | 读取逐帧 SMPL-X 参数（自动识别"每帧一个 pkl"和"每个 clip 一个 pkl"两种格式，并按帧号数值排序），完全复刻上游处理：根节点平移和朝向固定为最后一帧、下半身和脊柱固定、绕 x 轴转 180°（从相机坐标系转到 y-up）。然后在 GPU 上批量跑 SMPL-X 前向得到关节位置，输出身体动作和手部关节点。`--inspect` 只打印数据结构，不做处理 |
| `generate_motion_yaml.py` | 根据 `index.csv` 按帧数过滤，生成 motion yaml。其中 `root` 指向身体数据目录，`hand_root` 指向手部数据目录 |
| `retarget_hands.py` | 把人手 21 个关节点对齐到规范朝向（腕→中指根对准 +z，再绕 z 旋转），然后用 dex-retargeting 的 vector retargeting 优化 Linker 手的 15 个关节角，使指尖向量一致 |

### 1.3 `ASE/`（NVIDIA ASE 的 poselib，负责动作表示与身体重定向）

| 文件 | 作用 |
|---|---|
| `ase/poselib/poselib/skeleton/skeleton3d.py` | `SkeletonTree`、`SkeletonState`、`SkeletonMotion`：骨架和动作的数据结构；`retarget_to_by_tpose` 基于 T-pose 做重定向 |
| `ase/poselib/retarget_motion_h1_smpl.py` | **手语用的身体重定向**：14 个 SMPL 关节 → H1 的 14 个 link（关节映射见 `data/configs/retarget_smpl_to_h1.json`）。按 T-pose 计算相对旋转，再把根节点高度调整到双脚着地 |
| `ase/poselib/retarget_motion_h1_all.py` 等 | CMU 动捕使用的版本，上游 README 写错了，手语数据**不要用** |
| `ase/poselib/data/tpose/{smpl,h1}_tpose.npy` | 源骨架和目标骨架的 T-pose |
| `ase/poselib/data/configs/*.yaml` | motion 列表，新生成的 `motions_how2sign_*.yaml` 也放在这里 |
| `ase/utils/motion_lib.py` | **训练时的参考动作库**：加载 yaml 里的全部动作和手部数据，拼成一个大张量放在 GPU 上；`get_motion_state(ids, t)` 在两帧之间插值（身体用 slerp，手部用线性插值），返回根位姿、19 个身体 DoF、关键 body 位置、30 个手指角、4 个额外手腕角。首次加载后会缓存为 `data/pkl/<yaml>.pkl` |
| `ase/env/`、`ase/learning/`、`ase/run.py` | ASE 原有的 AMP/ASE 训练代码，SignBot **不使用** |

### 1.4 `legged_gym/`（IsaacGym 环境，基于 ETH legged_gym / Exbody）

| 文件 | 作用 |
|---|---|
| `legged_gym/__init__.py` | 定义 `LEGGED_GYM_ROOT_DIR` 和 `ASE_DIR` |
| `envs/__init__.py` | **任务注册**：`h1_mimic_hand`（训练）、`h1_mimic_eval`（play）、`h1_view`（keybody 生成和回放），对应的配置都是 `h1_mimic_linker_config.py` |
| `envs/base/base_task.py` | 创建 sim/viewer、键盘事件，提供 `get_observations` |
| `envs/base/legged_robot_hand.py` | **带手的 H1 基类**：加载 URDF `resources/robots/h1/h1_linker_extrawrist.urdf`（共 55 个 DoF），设置 PD 增益和域随机化（摩擦、质量、电机强度、动作延迟），实现通用奖励项（`alive`、`orientation`、`torques`、`feet_*` 等） |
| `envs/h1/h1_mimic_hand.py` | **核心训练环境 `H1MimicHand`**：动作库加载、观测构造、跟踪奖励、终止条件和课程（详见第 2 节） |
| `envs/h1/h1_mimic_linker_config.py` | **训练配置**：环境维度、奖励权重、PD 参数、motion yaml 名称、PPO 超参 |
| `envs/h1/h1_mimic_eval.py` | play 时使用的评估环境，增加录像相机 |
| `envs/h1/h1_mimic_view_motion.py` | `h1_view`：每个 env 对应一条 motion，纯运动学回放，记录 12 个关键 body 的局部位置，写入 `*_key_bodies.npy` |
| `envs/h1/h1_mimic.py`、`h1_amp.py`、`h1_command.py`、`*_config.py` 等 | Exbody 原有的不带手版本或 AMP 版本，手语训练不使用 |
| `envs/a1`、`anymal*`、`cassie`、`go1` | legged_gym 自带的其他机器人，不使用 |
| `scripts/train.py` | 训练入口：初始化 wandb，然后 `make_env` → `make_alg_runner` → `learn` |
| `scripts/play.py` | 加载 checkpoint 做可视化、录像，或导出 jit |
| `scripts/gen_keybody.py`（**新增**） | 无头模式生成 keybody 文件，不需要 dummy 模型 |
| `scripts/save_jit.py` | 把策略导出为 TorchScript |
| `utils/helpers.py` | 命令行参数（`--num_envs`、`--motion_name`、`--resume` 等）以及配置覆盖逻辑 |
| `utils/task_registry.py` | 根据任务名构造 env 和 runner，并处理续训时的模型加载 |
| `utils/terrain.py` | 地形生成；这里只用接近平地的小起伏地形（高度 0–4cm） |
| `resources/robots/h1/` | H1 机器人 URDF 和 mesh，其中 `h1_linker_extrawrist.urdf` 是 H1 加 Linker 手、带额外手腕自由度的版本 |

### 1.5 `rsl_rl/`（PPO 实现，Exbody 修改版）

| 文件 | 作用 |
|---|---|
| `runners/on_policy_runner_mimic.py` | `OnPolicyRunnerMimic`：采样 → 计算 GAE → PPO 更新 → 写日志和保存模型 |
| `algorithms/ppo_mimic.py` | `PPOMimic`：clip PPO，外加一个状态估计器（estimator）损失和隐变量正则项 |
| `modules/actor_critic_mimic.py` | `ActorCriticMimic`：策略网络和价值网络 |
| `storage/rollout_storage.py` | 存放 rollout 数据，计算 return 和 advantage |
| `modules/amp_discriminator.py`、`runners/*amp*`、`algorithms/{RPO,mdpo,ppo_penalty}_mimic.py` | 实验性的变体，默认配置不使用 |

### 1.6 `dex-retargeting/`（手部重定向库，作者加入了 Linker 手）

- `dex_retargeting/configs/teleop/linker_hand_{left,right}.yml`：vector retargeting 配置
  - 5 条"腕→指尖"向量（拇指、食指、中指、无名指、小指）
  - scaling 系数为左手 1.2、右手 1.45
  - 低通滤波系数 0.8
- `optimizer.py`、`seq_retarget.py`：基于 pinocchio 正运动学 + nlopt 的逐帧优化，每帧以上一帧结果作为初值（warm start）

## 2. 关键训练代码结构

### 2.1 一次训练迭代

```
OnPolicyRunnerMimic.learn()                          # rsl_rl/runners/on_policy_runner_mimic.py
  for it in range(max_iterations):                   # 默认 25000
      for step in range(24):                         # num_steps_per_env
          actions = actor_critic.act(obs)            # 11 维
          obs, rew, done, info = env.step(actions)   # H1MimicHand.step
          storage.add(...)
      alg.compute_returns(critic_obs)                # GAE: gamma=0.99, lambda=0.95
      alg.update()                                   # 5 epochs x 4 minibatches
      每 500 次迭代保存一次 model_<it>.pt
```

### 2.2 `H1MimicHand.step`（单个控制步 = 4 个仿真子步 × 5ms = 20ms，控制频率 50Hz）

1. **动作延迟随机化**：从动作历史缓冲中取第 `-delay-1` 帧的动作（训练时 delay ∈ {0, 1}）。
2. **下半身 PD 控制**：动作 `a ∈ R^11` 对应 左腿 5 个、右腿 5 个、躯干 1 个关节。
   - `τ = Kp · (0.25·a + q_default − q) − Kd · q̇`
   - Kp 和 Kd 按电机强度随机化 0.8–1.2 倍
3. 物理仿真 4 个子步。
4. `post_physics_step`：
   - 刷新状态，计算奖励，检查终止，重置需要重置的 env，构造观测。
   - 参考动作时间推进 20ms，更新 demo 观测。
   - **上半身用运动学方式直接设置**：把参考动作中的双臂 10 个 DoF（肩 3 个、肘 1 个、hand-roll 1 个，左右各一套）、4 个额外手腕 DoF 和 30 个手指 DoF 直接写入关节状态，等于每步"瞬移"过去。

> **这一点是理解 SignBot 训练的关键**：在当前代码中，**RL 只学习下半身和躯干的 11 个关节**，任务是在上半身做手语动作时保持平衡，并跟踪参考动作的根节点姿态和关键点。手臂和手指的动作由重定向直接给出，不经过策略网络。

### 2.3 观测（`compute_observations`）

**proprio（133 维）** 的组成如下，合计 3+2+2+55+55+11+2+3 = 133：

| 内容 | 维度 |
|---|---|
| 根节点角速度 | 3 |
| roll、pitch | 2 |
| sin/cos(yaw − 目标 yaw) | 2 |
| 关节位置（减去默认值） | 55 |
| 关节速度 | 55 |
| 上一步动作 | 11 |
| 足底接触（实际被置为常数） | 2 |
| 投影重力 | 3 |

**demo（98 维）** 来自参考动作的下一帧（未来 0.1s），合计 19+3+3+2+1+36+30+4 = 98：

| 内容 | 维度 |
|---|---|
| 身体 DoF | 19 |
| 局部线速度 | 3 |
| 局部角速度 | 3 |
| roll、pitch | 2 |
| 根节点高度 | 1 |
| 12 个关键 body 的局部位置 | 36 |
| 30 个手指角 | 30 |
| 4 个额外手腕角 | 4 |

**完整观测**按 `[proprio 历史 4 帧 | 当前 proprio | demo | 特权信息 | 10 帧历史]` 拼接。

- 特权信息：质量、摩擦、电机强度，代码中都乘了 0，实际为全零。
- 10 帧历史：同样被置零。

### 2.4 网络（`ActorCriticMimic`）

- **Actor**：
  1. 历史编码：proprio 历史 4 帧，每帧经过 MLP(133→128→16)，再拼接成 64 维，经 MLP 合并为 16 维。代码里这部分叫 `text_feat_encoder`，名字有误导性，实际处理的是本体感觉历史。
  2. 拼接：16 维历史编码 + 当前 proprio（133）+ demo（98），共 247 维。
  3. 主干：MLP [512, 256, 128] + ELU。
  4. 输出：11 维动作均值；标准差是可学习参数，初值 1.0。
  - 代码里还计算了特权编码和历史编码，但结果随后被覆盖，**没有参与前向**。
- **Critic**：完整观测 → MLP [512, 256, 128] → 状态价值 V。
- **Estimator**：从 proprio 预测基座线速度（3 维）。它的优化器那一步被注释掉了，加上特权输入本身为零，所以**实际上不起作用**。

### 2.5 PPO 超参（`legged_robot_config.py` 与 `h1_mimic_linker_config.py`）

| 项 | 取值 |
|---|---|
| clip | 0.2 |
| γ / λ | 0.99 / 0.95 |
| 学习率 | 2e-4，按 KL 自适应调整，目标 KL 为 0.01 |
| entropy 系数 | 0.005 |
| 梯度裁剪 | 1.0 |
| value loss | clipped |
| 每次迭代样本数 | 4096 env × 24 步 ≈ 9.8 万 |
| 更新 | 5 epochs × 4 minibatches |
| 最大迭代数 | 25000 |

### 2.6 奖励（权重见 `h1_mimic_linker_config.py → rewards.scales`）

**跟踪项（正奖励）**：

| 项 | 权重 | 形式 |
|---|---|---|
| `tracking_lin_vel` | 6 | `exp(−4‖v − v_demo‖)` |
| `tracking_demo_key_body` | 6 | `exp(−‖12 个关键 body 位置 − 参考位置‖)` |
| `tracking_demo_dof_pos` | 5 | `exp(−0.7‖q_19 − q_demo‖)` |
| `tracking_demo_yaw` | 1 | 朝向（yaw）跟踪 |
| `tracking_demo_roll_pitch` | 1 | roll/pitch 跟踪 |
| `alive` | 1 | 常数 |
| `feet_air_time` | 10 | 抬脚时长 |

**正则项（负奖励）**：

| 项 | 权重 |
|---|---|
| `orientation` | −5 |
| `dof_pos_limits` | −10 |
| `feet_stumble` | −2 |
| `lin_vel_z` | −1 |
| `ang_vel_xy` | −0.4 |
| `torque_limits` | −0.2 |
| `action_rate` | −0.1 |
| `dof_error` | −0.1 |
| `feet_drag` | −0.1 |
| `feet_force` | −3e-3 |
| `energy` | −1e-3 |
| `torques` | −1e-5 |
| `dof_acc` | −3e-7 |

奖励允许为负，并做了截断（`clip_rewards`）。

### 2.7 初始化、终止和课程

- **参考状态初始化（RSI）**：每个 env 固定绑定一条 motion（`env_id % 动作数`）。重置时随机抽取一个起始时刻，并把机器人设置成该时刻的参考姿态。
- **终止条件**：满足任一条即终止。
  - 躯干发生接触
  - `tracking_demo_dof_pos` 奖励低于 0.1，即关节偏差过大
  - 参考动作播放结束
  - 超时（50s）
- **课程**：根据每个 env 的动作完成率，自适应收紧或放宽 dof、keybody、yaw、height 四个偏差阈值，并记录到日志 `curriculum_*_term_thresh`。
- **域随机化**：
  - 摩擦、基座质量
  - 电机强度 0.8–1.2
  - 推搡（push）
  - 重力 ±0.1，每 10s 重新采样一次
  - 动作延迟 0–1 步
  - 观测噪声：noise_scale 0.5

## 3. 算法总结

SignBot 的运动控制部分可以概括为：**"重定向 + 运动学上半身 + RL 下半身平衡"的动作模仿（motion imitation）**，整体框架继承自 Exbody（Expressive Whole-Body Control）。

1. **离线重定向**
   - 人体 SMPL-X 动作通过 T-pose 相对旋转映射到 H1 的手臂关节。
   - 人手 21 个关节点通过 vector retargeting（优化指尖向量一致）映射到 Linker 手的 15 个关节。
   - 下半身固定为站姿模板。
2. **在线策略学习**：PPO 训练一个 11 维的下半身策略。
   - 输入：本体感觉、4 帧历史，以及未来 0.1s 的参考信息（关节角、根节点速度和姿态、关键点位置、手指角）。
   - 奖励：以跟踪奖励为主，加上大量平滑性和安全性正则。
   - 训练技巧：RSI、自适应终止课程、域随机化，用来提升鲁棒性和 sim2real 能力。
3. **部署**：上半身关节直接执行重定向得到的轨迹，下半身由策略闭环控制保持平衡。

论文中的 Translator（Uni-Sign）、Generator（SOKE，文本→手语动作）和 Responder（LLM）都在这条 RL 流水线之外：Generator 产出的 SMPL-X 动作可以接到第 0 节数据流的起点。
