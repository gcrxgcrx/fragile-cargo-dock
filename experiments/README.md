# CREATE vs EUREKA-style population search — FragileCargoDock-v0

一次同环境、同模型、同预算的奖励设计方法对照实验。

| 项目 | 设置 |
|---|---|
| 环境 | `FragileCargoDock-v0`（19 维观测 / 2 维连续动作 / 400 步） |
| 任务 | 把易碎货箱推过隔墙缺口，低速低冲击地精确送入泊位并稳定保持 |
| 模型 | `deepseek-flash`（即 DeepSeek-V4.1-Flash），两个方法相同 |
| 思考模式 | 关闭（`DEEPSEEK_THINKING=disabled`），两个方法相同 |
| 预算 | 各 10 次训练 × 300 万步 = **3000 万环境步**，两个方法相同 |
| 选择指标 | 环境原生目标 `external_eval.mean_eval_reward`（两方法一致） |
| 种子 | seed 0（单 seed，见"局限"） |
| 官方奖励 | 对两个方法**均隐藏**（`masked_step_source.py`），LLM 只能读到脱敏后的任务描述 |

**CREATE** = 单谱系 + 结构化诊断 + 语义局部化编辑 + 干预记忆 + best 归档
（`run_create_fragilecargo.sh` → `pipeline.run_iterative_experiment`）

**EUREKA-style** = 种群采样 + 训练评估 + 按原生分数选择 + 精英带过代 + 反射式自由重写
（`run_eureka_fragilecargo.sh` → `pipeline.run_eureka_population`）

---

## 1. 结果

| 策略 | 原生分数 | 任务成功率 | 预算 |
|---|---:|---:|---:|
| PPO + **官方原生奖励**（上界参考） | **299.65** | **96.8 %** | 300 万步 |
| 手写启发式控制器 | 158.41 | 50 % | — |
| **EUREKA-style（最优候选 g02c03）** | **+5.126** | **0 / 20** | 3000 万步 |
| **CREATE（最优轮次 iter_10）** | **+4.864** | **0 / 20** | 3000 万步 |
| 随机策略 | −1.14 | 0 % | — |

**结论：两个方法都没有解出任务（20 个评估回合全部超时，0 次成功入库），最终分数相差 0.26 分。**

---

## 2. CREATE 逐轮

| 轮 | 分数 | 终止 | 干预层级 | 主导组件（份额 / 激活率） |
|---|---:|---:|---|---|
| 01 | −2.263 | 0/20 | — | `crate_settling_and_alignment` **100 % / 100 %** |
| 02 | +1.614 | 0/20 | Level 2 | settle 96 % / 63 %　`crate_to_dock_progress` 3 % / **42 %** |
| 03 | **+3.861** | 0/20 | Level 2 | settle 70 % / 39 %　progress **26 %** / 34 % |
| 04 | −1.535 | 0/20 | Level 3 | `joint_dock_completion` **100 % / 100 %** |
| 05 | −1.687 | 0/20 | Level 3 | `dock_completion_state` **100 % / 100 %** |
| 06 | −6.736 | 1/20 | Level 3 | `joint_dock_completion` **99 % / 100 %** |
| 07 | −6.059 | 1/20 | Level 3 | `joint_dock_completion` **100 % / 100 %** |
| 08 | +3.508 | 0/20 | Level 3 | progress 68 % / 45 %　joint 31 % / 100 % |
| 09 | −1.939 | 0/20 | Level 3 | `dock_completion_joint` **100 % / 100 %** |
| 10 | **+4.864** | 0/20 | Level 3 | `dock_progress_delta` **81 % / 81 %**　gate 12 % / 80 % |

「份额」= 该组件绝对均值占全部组件绝对均值之和的比例。**单一组件份额 100 % 即结构塌缩：策略只需满足那一个组件即可刷满奖励。**

### 发现 A：Level 2（定点修一个组件）有效，Level 3（重建骨架）低效

- **Level 2 共 2 轮，2/2 都拿到正分并刷新 best。** 第 2、3 轮是教科书式的语义局部化修复：诊断出「静止即得分」的刷分路径 → 只给 settle 项加进度门控 → 再把状态值改成改善量并压低权重。结构从 **settle 100 % 单一垄断**逐步再平衡到 **settle 70 % / progress 26 %**，主信号激活率从 **0.2 % 提升到 42 %**。
- **Level 3 共 7 轮，只有 2 轮（08、10）拿到正分**，其余 5 轮全部塌缩成**新的单组件 100 % 份额刷分坑**（04、05、06、07、09）。

也就是说：**诊断是准的，但"重建骨架"这个动作反复把有效结构一起丢掉，再自由生成时又掉进同类陷阱。**

### 发现 B：诊断给出的定量预测可以被逐轮核对

第 2 轮反思代理给出了可证伪预测（`crate_to_dock_progress` 激活率 > 30 %、settle 份额 < 70 %、分数上升）：

| 预测 | 实际（iter_02） | |
|---|---|---|
| progress 激活率 > 30 % | **42 %**（从 0.2 % 提升约 200 倍） | 命中 |
| settle 份额 < 70 % | 96 % | 未兑现 |
| 分数上升 | −2.26 → +1.61 | 命中 |

未兑现的那一条，在**下一轮**被补上了（settle 份额 96 % → 70 %）。每轮的 8 字段输出（`evidence` / `behavior_diagnosis` / `signal_completeness` / `selected_level` / `selected_intervention` / `falsifiable_hypothesis` / `expected_next_round` / `main_risk`）完整保存在 `create/diagnoses/`。

---

## 3. EUREKA-style 逐代

| 代 | 候选 | 类型 | 父代 | 原生分数 |
|---:|---|---|---|---:|
| 0 | g00c00 | 初始采样 | — | −102.241 |
| 0 | g00c01 | 初始采样 | — | −1.863 |
| 0 | g00c02 | 初始采样 | — | −7.008 |
| 0 | g00c03 | 初始采样 | — | −2.473 |
| 1 | g01c02 | 子代编辑 | g00c01 | −1.340 |
| 1 | g01c03 | 子代编辑 | g00c03 | +3.550 |
| 2 | g02c02 | 子代编辑 | g01c03 | +3.377 |
| 2 | g02c03 | 子代编辑 | g01c02 | **+5.126** |
| 3 | g03c02 | 子代编辑 | g02c03 | +3.455 |
| 3 | g03c03 | 子代编辑 | g02e00 | +2.419 |

（精英带过代条目略，见 `eureka_generations.csv`）

**发现 C：EUREKA 的最优候选来自编辑「较差的那个精英」。** 第 2 代的两个精英是 g01c03（+3.55）和 g01c02（−1.34）；编辑更好的 g01c03 得到 +3.38（变差），编辑更差的 g01c02 得到 **+5.126**（全局最优）。这提示"按当前分数选精英、只编辑最优者"并非可靠策略。

**第 3 代没有再提升**，搜索在 ~+5 附近收敛。

### 两个方法都没识别出的同一件事

EUREKA 最优候选在训练末期各组件激活率都在上升（`crate_dock_alignment` 59.9 % → 91.5 %、`crate_settling` 20.4 % → 79.4 %、`joint_condition_proxy` 10.9 % → 71.8 %），**奖励被优化得越来越充分，任务成功率始终为 0**。

CREATE 的反射里带**结构化的完整性审计**（份额 + 激活率 + 与外部分数的对照），因此它能在第 2 轮就定位到「主信号激活率 0 %，即不可达」；EUREKA 的反射同样含组件数值，但没有这个审计框架，4 代 10 个候选都没把「货箱根本没动」作为诊断结论。

---

## 4. 局限（读这份结果前必须知道）

1. **地板/天花板效应。** 两个方法**都是 0/20 成功**。分数差 0.26 分落在噪声区间内，**不能据此判断哪个方法更好**。
2. **单 seed。** 每方法只跑了 1 次完整搜索，`population_size=4` 的 best-of-4 方差很大（本轮初始 4 个样本里就有 1 个是 −102 的自毁奖励）。
3. **任务对"从描述设计奖励"整体偏难。** 官方原生奖励能到 96.8 %，但把它隐藏、只给文字描述，两个方法在 3000 万步内都触底。任务的关键技巧是**小车无法给货箱刹车，必须提前松手让货箱靠阻尼滑入泊位**——这一点很难从文字描述推断。

**因此本实验目前支持的是机制层面的结论（诊断可归因、干预可核对），而不是"CREATE 分数更高"。**

---

## 5. 文件索引

```
experiments/
├── README.md                      本报告
├── create_rounds.csv              逐轮：分数 / 终止 / 各组件份额与激活率
├── eureka_generations.csv         逐候选：代 / 类型 / 父代 / 分数
├── create/
│   ├── experiment_summary.md      原始汇总（含 stopped_reason）
│   ├── reward_memory.md           干预记忆表（10 轮决策）
│   ├── best_reward.py             最优奖励（iter_10, +4.864）
│   ├── diagnoses/iter_XX_diagnosis.md   每轮 8 字段反思输出（9 份）
│   └── rewards/iter_XX_reward.py        每轮生成的奖励代码（10 份）
└── eureka/
    ├── eureka_summary.md          原始汇总
    ├── best_reward.py             最优奖励（g02c03, +5.126）
    └── rewards/gXXcYY_reward.py   各候选奖励代码（10 份）
```

未包含：`model.zip`、`vecnormalize.pkl`、训练 monitor 日志、完整 prompt 记录（约 10 MB/方法，可由配置重跑复现）。

---

## 6. 复现所需的代码位置

本仓库只包含环境与基线；两个奖励搜索方法属于 `expert-reward-agent` 主管线，对应文件为：

| 文件 | 作用 |
|---|---|
| `pipeline/run_eureka_population.py` | EUREKA-style 种群搜索（本仓库外） |
| `pipeline/run_iterative_experiment.py` | CREATE 单谱系编排器（本仓库外） |
| `prompts/eureka_01_initial_reward.md` | 初始奖励生成 prompt |
| `prompts/eureka_02_reward_edit.md` | 基于 reward reflection 的编辑 prompt |
| `configs/env007_fragilecargo_eureka_baseline.yaml` | EUREKA 配置 |
| `configs/env007_fragilecargo_eureka.yaml` | CREATE 配置 |
| `run_eureka_fragilecargo.sh` / `run_create_fragilecargo.sh` | 两个方法的进程文件 |
| `training/train_sb3_wrapper.py` | 内层 PPO 训练（本实验为其增加了 `monitor_snapshots`） |

这些文件依赖主管线的 `prompts/`、`rag/`、`knowledge_base/` 等目录，单独放进本仓库无法直接运行，故未纳入。
