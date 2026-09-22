# ChatGPT's answers to P1–P5 (verbatim archive)

Archived 2026-09-20. Two separate replies from the same external model, merged here so
the set is complete: P3 and P5 came first, P1/P2/P4 second. The companion DeepSeek-V4.1F
answer is `deepseek_v41f_answers_P1-P5.md` in this directory.

**My adjudication of both is in `../../SESSION_STATE.md` section 5 and in the conversation
summary. The decisive fact neither could verify — that EUREKA's reflection already
contains per-component activation rate — is in `SESSION_STATE.md` section 3g.**

---

# Direct answers

## P3

否。`mass of the initial-reward distribution near the threshold` 不是 Condition C 的 predictor；它衡量的是 initial sampling 已经撞上好 reward 的概率，因而首先预测 best-of-\(k\) population sampling 的收益，甚至可能与 CREATE 的相对优势方向相反。Condition C 的正确对象是 standard decision theory 中的 **expected value of sample information (EVSI)**（样本信息对“选择下一次 reward edit”这一决策的期望价值）：在控制 EUREKA 已有证据 \(E_0\) 后，CREATE 独有证据 \(E_1\) 能否改变 edit 选择，并提高同预算下的 downstream utility。操作上，应在固定 naive reward、固定 LLM、固定 edit budget 下，比较 \(E_0\) 与 \((E_0,E_1)\) 所选择 edit 的 paired utility；当 `95% lower confidence bound` 同时满足 \(\widehat{EVSI}(E_1\mid E_0)>0\) 且 \(V_{\text{single-lineage}}(B)-V_{\text{population}}(B)>0\) 时，才预测 CREATE 胜出。现有 LunarLander / BipedalWalker / FragileCargoDock 三个 environment 不能检验这个 environment-level predictor；五个 seed 或几十个 reward 不能把 \(n=3\) 变大。应建立预注册的 procedural factorial benchmark，保留 C-positive、C-neutral、C-negative 全部 strata，并用 held-out task families 检验 method × EVSI interaction，而不是专门筛选“适合 CREATE”的 environment。

## P5

**it does not exist**：不存在一个仅修改 `FragileCargoDock-v0` mechanics、同时又能保证 Condition C 的 minimal change。原因不是物理机制不够复杂，而是 C 是 evidence-channel-relative property；当前 EUREKA implementation 已收到 `mean_episode_length`、`termination_breakdown`、per-component mean / abs mean / min / max，以及随训练变化的 `active_rate`，所以把失败改造成“已有 component 权重错误”至多让两个方法都更容易诊断，不能保证 CREATE 独占信息。正确动作是保持 environment 不变，把 CREATE 的 evidence channel 扩展为 trajectory-conditioned counterfactual evidence（按 success predicate transition 对 trajectory 分段，并报告 event-conditioned component contributions 与 reward sensitivity），同时冻结 EUREKA channel。这样缺失的 closing-speed incentive 会以“进入 dock 时速度条件反复失败，但 reward 对 reachable slow/fast trajectory contrast 不敏感”的形式出现；这改变的是可观察信息，而不是为 CREATE 改造任务。

# Detailed argument

## P3

### Why threshold mass is the wrong quantity

首先，它没有识别 C。可以构造两个 environment，initial-score distribution 完全相同：一个失败来自现有 `stability` component 权重过小，另一个失败来自完全缺失的 braking signal。两者的 near-threshold mass 相同，但前者可能被 component evidence 定位，后者不能。反过来，只改变 EUREKA/CREATE 的 evidence presentation，C 会改变，而 initial scores 完全不变。

其次，它不是稳健量。对 native score 做不影响 policy ordering 的 affine transformation，或者改变一个人为 success threshold，near-threshold mass 就会改变。最重要的是，高 upper-tail mass 通常意味着多抽几个 independent candidates 很可能成功——这正是 Eureka 所采用 evolutionary population search 能利用的资源，而不是 diagnosis 的独占优势。因此 BipedalWalker 的 `97 %` 不能作为 CREATE 优势的正证据。

正确形式来自 Blackwell (1951) 的 **Blackwell informativeness**（信息结构是否使某个决策问题取得更低 Bayes risk）以及 Lindley (1956) 的 information value。令 \(E_0\) 为 EUREKA 实际收到的 evidence，\(E_1\) 为 CREATE 额外收到的 evidence，\(A(r)\) 为对 naive reward \(r\) 合法的 L1/L2 edit 集合，\(Y(r,a)\) 为执行 edit \(a\) 后的目标 utility，则：

\[
EVSI(E_1\mid E_0)=
\mathbb E\!\left[\max_a\mathbb E[Y(r,a)\mid E_0,E_1]\right]
-
\mathbb E\!\left[\max_a\mathbb E[Y(r,a)\mid E_0]\right].
\]

这是 C 应当对应的量。实际 LLM 并非 Bayes-optimal decision maker，因此还应报告 fixed LLM decision rule 的 realized value：

\[
\widehat{\Delta}_{C}
=
\frac1N\sum_{i=1}^{N}
\left[
Y(r_i,a_i(E_0,E_1))-Y(r_i,a_i(E_0))
\right].
\]

这一区分很重要：\(E_1\) 理论上更 informative，不代表当前 LLM 能使用它。

### Operational procedure

1. 冻结 generator model、temperature、prompt、reward wrapper、edit taxonomy 和 training budget；先生成至少 32 个 naive rewards，不能根据 score 筛选。
2. 建立 64 条 physically reachable trajectory 的固定 panel。对 `FragileCargoDock-v0`，必须覆盖给定的 `0.2, 0.3, 0.5, 1.0, 2.0 m/s` release regimes，并包含 success、5–9 stable steps、overshoot、parked-outside、hard-impact 和 out-of-bounds。用 environment 的真实 evaluation objective \(J(\tau)\) 给 trajectory 排序，但不向 reward-generating LLM 暴露 native reward code。
3. 对每个 naive reward，在完全相同的 trajectory 上生成 \(E_0\) 与 \(E_1\)。随后以 matched random seeds 让同一个 LLM 分别根据 \(E_0\) 和 \((E_0,E_1)\) 选择一个 edit。不要让两个 arm 使用不同 prompt advice。
4. 无训练阶段计算 `pairwise ranking accuracy`：对所有 \(J(\tau_i)>J(\tau_j)\) 的 trajectory pairs，检查 edited reward return 是否也满足 \(R(\tau_i)>R(\tau_j)\)，tie 计 `0.5`。它只能验证 structural diagnosis，不能验证 learnability；附件中的 v4 gate 已经直接证明“静态检查通过”不蕴含 trained policy 成功。
5. cheap-training bridge 使用完整预算的 `10 %`，即每 candidate `0.3M` steps，并在 `0.05M, 0.1M, 0.2M, 0.3M` 做 fresh evaluation，以 native-score learning-curve AUC 为 \(Y\)。同一个 initial reward、policy seed 和 evaluation seeds 必须 paired。另保留 `20 %` 的 cases 跑到 `3.0M`，检验 early AUC 是否真的预测最终 fresh success；若不预测，就不能用 cheap proxy 宣称方法胜负。
6. 在总训练预算 \(B\) 下，population comparator 必须使用相同数量的 `0.3M` candidate trainings。决策规则不是任意设一个 “EVSI > 0.1”，而是预注册：

\[
LCB_{95\%}(\widehat{\Delta}_{C})>0
\quad\text{且}\quad
LCB_{95\%}\!\left(
V_{\text{single-lineage}}(B)-V_{\text{population}}(B)
\right)>0.
\]

bootstrap 必须以 environment/task 为最高层 resampling unit，其次才是 reward 和 seed。

现有三环境数据不能检验这个规则。predictor 在 environment level 只有三个点；把同一 environment 的 seeds 当成独立 task 是 **pseudoreplication**（Hurlbert 1984）。若期望检测约 \(r=0.6\) 的强关系，粗略也需要约 20 个 independent tasks；三点只能画图，不能验证 predictor。

### Constructing environments without designing for CREATE

不能彻底消除该 threat，但可以把它从“选择性设计”降为“预注册的 diagnostic benchmark”。具体做法是先冻结一个 method-independent task grammar，再采样完整 factorial family：动力学参数、几何容差、contact damage、dwell requirement、actuation limits 分别独立变化；同时对 reward defects 预先规定 `coefficient rescaling`、`component deletion`、`spurious proxy` 三类 transformation，全部保留，不按 CREATE 表现筛选。task generator、hidden seeds、defect distribution 和 analysis rule 在运行 CREATE 前公开哈希；开发 instances 与 held-out families 分离。最终报告 method × measured-EVSI interaction，而不是只报告 C-positive subset。

这与 Cobbe et al. (2020) 用 procedural distributions 检验 generalization，以及 Osband et al. (2020) 用预定义 diagnostic experiments 分离 capability 的原则一致。它仍不能证明现实任务中 C 的 prevalence；因此 custom family 的结论必须限定为“在预注册的 defect distribution 上，额外 evidence 有 decision value”，不能写成无条件 superiority。

### Falsifiable prediction

在至少 20 个 held-out tasks 上，\(\widehat{EVSI}(E_1\mid E_0)\) 应与 CREATE-minus-population 的 budget-matched performance gap 同号，并在 hierarchical analysis 中显著解释该 gap；near-threshold mass 在控制 EVSI 后不应保留稳定预测力。对 `FragileCargoDock-v0` 当前 aggregate evidence，\(\widehat{EVSI}\) 应接近 `0`。

### Defeat condition

如果在预注册的 held-out family 上，near-threshold mass 在控制 EVSI 后仍稳定预测 method gap，而 \(\widehat{EVSI}\) 不预测；或者 richer evidence 以显著正 EVSI 选出更好的 edit，却没有提高任何 full-budget outcome，那么我的 predictor 被推翻。

---

## P5

### Why no environment-only minimal change can create C

代码层面的关键事实是：`run_eureka_population.py:175–260` 不只是给 score。它已经给 `mean_episode_length`、`termination_breakdown`、training return curve、per-component episode means、`active_rate` trajectory 和最终 `mean / abs_mean / min / max`。CREATE 的 `magnitude_share` 多数情况下只是 `abs_mean` 的 normalization，`signed_share` 也可由已有 component means 近似重建。CREATE 真正额外的是 cross-iteration delta、memory、investigator interpretation 和 localized-edit policy，而不是一张 EUREKA 原理上看不到的物理量表。

因此，即使加入一个“天然要求 safety–speed trade-off”的 mechanic，并使所有 naive rewards 都写出 safety component，weight imbalance 也会同时显现在两个 channels。它可能提高两个方法，但没有产生 C。environment cannot decide which method sees a statistic。

### Concrete evidence extension

应从 environment 的 atomic termination predicates 自动生成 event trace；不要把 `−0.05 · contact · max(...)` 或任何 coefficient 放进 prompt。对每一 episode，CREATE evidence 增加以下具体字段：

1. `episode_id`, `native_score`, `elapsed_steps`, `termination_reason`, `is_success`。
2. 对 `cargo_inside_dock`, `heading_ok`, `speed_ok` 分别记录 `first_true_step`, `last_true_step`, `longest_true_run`, `false_to_true_count`, `margin_at_break`；另记录 `max_stable_steps`。这会直接暴露“30/60 达到 5–9 steps，但 speed predicate 打断”。
3. 在 `first_dock_entry`、`max_stable_steps`、`dock_exit`、termination 前后各 `8` steps，记录 `obs[0:19]`、action、`cargo_speed`、`cargo_angle_error`、`contact`、`cart_forward_speed`、`cargo_velocity_along_cart_heading`、二者差、`hard_collision_count`。
4. 对每个 reward component，在上述 event windows 分别记录 `sum`, `abs_sum`, `active_rate`, `magnitude_share`，而不是只给整 episode aggregate。
5. 用 simulator-restored、physically reachable paired rollouts 做 **counterfactual trajectory evaluation**：保持 release position 与 heading 匹配，改变 release speed；输出每条 trajectory 的 `raw_return`, `clipped_return`, per-component return，以及 reward 是否正确排序 slow-success / fast-overshoot。再在 near-dock states 上报告对 `obs[4]`, `obs[8]`, `obs[9]`, `obs[12]`, `obs[13]`, `obs[14]` 的 finite-difference reward sensitivity。

这不是凭空发明的方法。利用 hypothetical behavior 主动检验 reward information 与 Reddy et al. (2020) 的 ReQueST 一致；用 event/predicate structure 分解 temporally extended task 与 Toro Icarte et al. (2018) 的 Reward Machines 一致；用 original/counterfactual partial trajectories 解释 reward 的直接先例是 Wehner et al. (2024)。

在当前失败 reward 上，这张 evidence 应呈现：`cargo_inside_dock=True`，但 `speed_ok=False`；`max_stable_steps` 集中在 `5–9`；高速度进入 dock 的 trajectory 与低速度成功 trajectory 得到近似相同 reward ordering；near-dock reward 对 cargo-velocity perturbation 的 sensitivity 近零。一个 aggregate component table 无法指出缺失变量，而这些字段能指出“哪一个 success predicate 在什么 event 被什么 state dimension 打断”。

### Distinguishing real C from “both methods got easier”

冻结 environment、initial rewards、LLM、prompt、policy seeds 和 EUREKA reflection，只对 CREATE 做三臂 paired ablation：现有 aggregate evidence、真实 trajectory table、将 trajectory table 在 rewards 间随机 permutation 的 sham control。真实 table 必须在 score 尚未改善之前，提高“生成包含 speed-sensitive term 的 edit”的概率；之后才应看到 median dock-entry speed 下降、`max_stable_steps` 达到 `10`、fresh success 上升。若 aggregate arm、真实 table arm、sham arm 同样改善，改善来自普通 LLM variance 或 shared prompt，不是 C。若 EUREKA 在冻结 channel 后以相同比率自行添加同一 term 并取得相同 success，C 也不存在。

### Conditions A and B

这项改变不触碰 environment 或 reward search topology，因此不会破坏 A/B；但必须强调，附件并未证明 A/B 已成立。probe D 证明“存在一个 human-known one-component edit”，没有证明该 edit 在 CREATE 的 proposal distribution 下可达；v4 已经证明“component 被写出来”也不等于 weight、boundary guard 和 terminal structure 同时正确。B 同样需要从同一个 starting reward 做 localized edit 与 unconstrained rewrite 的 paired comparison，不能由 reward 看起来含有多个合理 component 推断。

### Falsifiable prediction

加入真实 trajectory evidence 后，CREATE 应在 score 提升之前明确诊断 `speed_ok` 是稳定 streak 的主要 break predicate，并生成对 reachable fast/slow dock-entry trajectories 有显著 reward contrast 的 edit；aggregate-only 与 permuted-table arms 不应出现同等变化。随后 dock-entry speed、stable streak 和 fresh success 应按该因果顺序改善。

### Defeat condition

如果 permuted trajectory tables 与真实 tables 同样有效；如果新 evidence 只提高 reward score，却不改变 speed sensitivity、dock-entry speed 或 stable streak；或者冻结 channel 的 EUREKA 同样稳定发现并修复缺失 term，那么 trajectory extension 没有产生 Condition C，我的结论错误。

# Adversarial cross-examination

我认为最可能错误的 claim 是 P3 中“**A and B probably hold**”，其依据是“probe D 是 control v1 加一个 component，因此 exactly an L2 localized edit”。这混淆了 **edit distance** 与 **search reachability**。A 要求 good reward 在实际 proposal distribution、实际 evidence 和实际 LLM decision rule 下能通过 small edits 到达；一个研究者已经知道答案后写出的单 component patch，只证明 solution 在代码空间中距离近。v4 的 `15/16` rule compliance 但三个 trained passers 全部 `0/60`，反而说明“写出该 component”与“形成可学习的正确 reward basin”之间仍有巨大断层。B 也没有被直接验证：必须从相同 starting rewards 比较 localized edit 与 unconstrained rewrite 对已有正确 components 的保存率和最终 utility。

什么会证明我错：在至少 24 个未筛选 naive rewards 上，不提供 closing-speed formula，只提供预注册 trajectory evidence；如果 CREATE 在超过一半 starting rewards 上独立提出 speed-sensitive localized edit，并且 paired fresh evaluation 中 localized arm 同时显著优于 unchanged reward、unconstrained rewrite 和 budget-matched independent sampling，那么我接受 A/B 已被支持，而不再把 probe D 视为 human-oracle artifact。

# Confidence self-assessment

P3：**4/5**。论证基础强，Blackwell/Lindley 给出了正确的 decision-theoretic object，三环境不足也是明确的 experimental-unit 问题。扣分来自 **insufficient literature**：尚无已发表工作直接证明 EVSI 能预测 LLM-driven reward-search algorithm ranking，必须靠你的 benchmark 校准。

P5：**4/5**。environment-only change 不能保证 method-relative information asymmetry，这一点论证充分；代码检查也显示 EUREKA channel 比 P5 的文字假设更丰富。扣分同样主要来自 **insufficient literature**：counterfactual trajectory evidence 有相关先例，但尚无工作证明它能让 LLM reward repair 在这个连续-control failure 上可靠提出缺失 component。

如果只能做一件事推进项目：在一周内实现上述 event-conditioned trajectory table，并对同一批 32 个未筛选 naive rewards 做 `aggregate / real-table / permuted-table` blinded paired diagnosis test，以“是否在训练前提出 speed-sensitive edit”为预注册 primary endpoint。

---

# Direct answers (second reply)

## P1

`counterfactual reward probing` 是正确的诊断原语，但不是完整的一般答案。更成熟的表述是：在 **dynamics-aware coverage distribution（服从环境动力学的覆盖分布）** 上，对候选 reward 进行 **trajectory-preference validation（轨迹偏好验证）** 与 **occupancy/feature discrepancy analysis（访问分布或特征期望差异分析）**。`analyze_terminal_dominance.py` 已经证明这种思路有信息量：旧 spec 的 “reward at the dock ÷ parked outside” 只有 **1.02–1.24**，pilot 为 **172–5123**；相应 episode-credit ratio 是 **0.025–0.031** 对 **0.289–1.278**。但 synthetic state 既可能是 dynamics-out-of-distribution，也不能说明 reward 是否在整条轨迹上把 `−0.05` gentleness signal 淹没。推荐把 CREATE 的 evidence 扩展为对可达、成对匹配轨迹的完整 return ordering、行为指标和局部敏感度；不要把同一扩展给 EUREKA，否则它不再构成 evidence-asymmetry 实验。probe D 的 **73.3%** 表明 closing-speed information 确实足够，但不意味着某个固定公式必须进入 prompt。

## P2

可执行判据是：一个 clause 只有在它陈述环境、API、evaluator 或资源约束的事实，并且不排除任何能够在外部 task metric 上成功的 reward 实现时，才属于 legitimate interface specification；只要它约束 reward 的内部表示、时间分解、特定 state/trajectory ordering、feature、event、公式或 coefficient，就是 solution-bearing scaffold，应从 shared prompt 移除。按此判据，clip 和 termination 是 (a)；“docked 必须高于 parked”、one-off terminal event、`−0.05 · contact · max(0, v_cart − v_crate)` 都是 (b)。后两者显然泄露机制；前者虽然比公式抽象，却仍排除了可能通过 potential-based shaping 或长期 return 实现同一成功策略的 reward。该现象不应称为传统意义上的 `prompt leakage`；最准确的检索词是 `target leakage`、`evaluation contamination` 和 `scaffold confounding`。精确到“shared scaffold 使两个方法输出收敛、从而消灭 treatment contrast”这一形式，目前没有公认的单一专名。

## P4

不能把当前 gate 挽救成 training-free 的正向 selector；应把它降格为高召回率的 rejection filter，再用约 **3–10%** full budget 的 multi-fidelity training 选正例。它**正确拒绝**了两个后来为 **0/60** 的候选，却也**接受**了三个后来为 **0/60** 的候选，因此目前没有 positive predictive power。scripted-controller returns 和 reference-trajectory ranking 会修复两个已知漏洞：完整累计 return 能发现 `−0.05` 被 progress reward 淹没而以 **1.555 m/s** 入 dock，也能用 out-of-bounds 轨迹淘汰没有 boundary guard 的候选；但固定轨迹库无法判断 PPO 是否能学到该行为。鉴于 CREATE 从 training-seed **+17.694** 跌到 fresh-60 **+7.716**，而 naive initial reward 仅约为 threshold 的 **≈2%**，正向选择必须包含未复用 seed 上的短训练行为指标，而不能继续依赖 `mean_eval_reward` 或静态 gate。

# Detailed argument (second reply)

## P1 — Diagnosing an absent incentive

“缺项不可见”需要再精确一层：它对 reward-component table 不可见，但对“候选 reward 在哪些可达行为之间不能给出正确偏好”并不可见。只看已有 component 的 mean、abs_mean、nonzero_rate 与 magnitude share，当然不可能直接看到不存在的 closing-speed term；可是若有两条除 closing speed 外尽可能匹配的轨迹，候选 reward 对它们的累计排序就提供了可检验信号。这与 Abbeel and Ng 的 feature-expectation matching 属于同一思路；但它也继承同一个限制——如果 feature basis 与 probe coverage 中都没有 closing speed，就不存在凭空识别它的定理。

因此，不应把 `analyze_terminal_dominance.py` 称为一般意义上的 `counterfactual off-policy evaluation`。它更接近 reward-function validation。真正的 counterfactual OPE 通常估计“换一项 policy intervention 后会发生什么”；这里 reward 已知，问题是它是否在一组行为上表达正确偏好。Gleave et al. 的 EPIC 直接比较 reward、避免训练 policy，但 Wulfe et al. 指出在物理上不可能的 transition 上评价 reward 会产生 OOD 失真，并据此提出 DARD。因此这里应优先使用 scripted controllers 产生的 reachable transitions，而不是任意拼接 synthetic states。

每条 trajectory 应保存 controller family、controller parameters、seed、initial condition、reachability provenance、episode length、termination reason、success、dock-entry speed、minimum dock speed、maximum settled streak、time below `0.05 m/s`、out-of-bounds、hard-impact count，以及首次 contact、release、dock entry、settling、termination 的 phase boundaries。随后对每个候选 reward 顺序 replay，记录 unclipped return、经过实际 `±20` wrapper 后的 discounted return、各 phase return，以及 task outcome。

对匹配轨迹对 \((\tau_i,\tau_j)\)，保存 task-oracle preference \(y_{ij}\)、候选 reward margin \(m_{ij}(R)=G_R(\tau_i)-G_R(\tau_j)\)，并把 \(y_{ij}m_{ij}(R)\le 0\) 的 pair 作为具体 counterexample。还应保存 controller sweep 上的局部斜率，例如 return 对 dock-entry speed、release position、settled steps、impact count 的变化；以及“candidate-return 最大的 scripted trajectory”相对于“task metric 最优 trajectory”的 outcome regret。

低成本路径是只生成一次 trajectory library：围绕 release speed `0.2, 0.3, 0.5, 1.0, 2.0 m/s`、ballistic release position、横向偏差和 boundary-directed controls 做小规模 sweep。之后每个 reward 只需 replay 这些 observation sequences；300 条、每条最多 400 steps，也只有 120,000 次 reward call。

这项 evidence 应只进入 CREATE 的 method-specific channel；EUREKA 保持其原有 population reflection。论文 claim 也必须相应收窄为“CREATE with structured trajectory counterexamples versus EUREKA-style score/component feedback”，并把生成 library 的 environment steps 计入资源报告。

**Falsifiable prediction.** 在未见过、且没有共享 gentleness snippet 的候选上，缺少有效 closing-speed incentive 的 reward 会在 fast-entry versus gentle-entry 的 matched pairs 中产生非正 task-aligned margin；probe D 应在至少 90% 的 held-out matched pairs 上给出 gentle-entry 更高的累计 ordering。

**Defeat condition.** 如果缺项候选与 probe D 在 held-out reachable pairs 上的 ordering、局部斜率和 candidate-optimal scripted behavior 不可区分，或者这些字段加入 CREATE 后仍不能提高其发现 closing-speed mechanism 的比例，那么该 evidence construction 没有比现有 channel 增加可操作信息。

## P2 — An executable screening procedure

对每个准备加入 shared prompt 的 clause，执行以下程序：

1. **Source test。** 它是否只是陈述可由 simulator、wrapper、API 或 evaluator 直接验证的事实？若是，例如 reward signature、observation channel、`±20` clipping、success 后立即 termination，则暂定为 (a)。
2. **Successful-reward counterexample test。** 尝试构造一个在外部 task metric 上成功、但违反该 clause 的 reward。若存在，则 clause 并非任务成功的必要条件，而是在选择 reward representation，归为 (b)。这一步必须按 induced behavior 判断，而不是按 reward code 长相判断。
3. **Ordering test。** clause 是否指定某两个 feasible states、trajectory segments 或完整 trajectories 的 reward ordering？若是，它是 supervision 或 hidden evaluation criterion，不是 interface；放进 generation prompt 时归为 (b)。
4. **Representation test。** clause 是否指定 feature、component、one-off event、temporal placement、公式、coefficient 或 code skeleton？除非 API 本身强制要求，否则归为 (b)。
5. **Scaffold-ablation test。** 用同一 model 和 seed 分布，对“有 clause / 无 clause”各生成一批候选。若 clause 显著降低候选间结构方差，或使两种方法复制相同 mechanism，它就是具有因果效应的 scaffold treatment，不能藏在 shared layer 中。

据此，五个 counterexample 的分类如下。

- “the per-step reward is clipped to ±20”：(a)。这是 wrapper 对所有 reward 的外生变换；不知道它会产生错误实现，但它没有规定该如何解决 docking。
- “the episode terminates on success”：(a)。这是 transition/horizon semantics；尤其影响 one-off state detection 的可实现性，但没有规定必须用 terminal event。
- “the reward must rank docked above parked on scripted states”：(b)。它适合做 private evaluator，却不是 interface。一个通过 dense shaping、potential-based shaping 或不同 temporal decomposition 产生正确成功策略的 reward，未必在给定两个 scripted states 上满足该点对点排序。
- “use a one-off terminal event to signal completion”：(b)。它指定 temporal representation，并排除成功后 dense stream、pre-terminal shaping 等行为等价方案。
- “the gentleness term has the form `−0.05 · contact · max(0, v_cart − v_crate)`”：(b)。它同时泄露 feature、interaction structure、sign、hinge form 与 coefficient，已经基本给出 probe D。

prompt 应分成三层。shared layer 只放 observation/action schema、单位与边界、wrapper clipping、termination/failure semantics、合法依赖、代码签名、训练预算以及最终外部 success metric。不得放 reward ordering、component checklist、event skeleton 或公式。CREATE-specific layer 放其 localized-edit protocol、component evidence 和 P1 的 trajectory counterexamples；EUREKA-specific layer 放其 population generation、selection 和原有 reflection。closing-speed formula 与 one-off-event skeleton 只能进入独立的 `oracle-scaffold positive control`，不能进入主 comparison。

主实验最好采用 `method × scaffold` 的 factorial ablation：CREATE/EUREKA × interface-only/oracle-scaffold。interface-only cell 是论文主结论；oracle-scaffold cell 只回答“显式答案是否使方法差异消失”。

术语上必须小心。Agarwal et al. 所说的 `prompt leakage` 是隐藏 system prompt 被模型泄露给攻击者，并不是这里的现象。Kaufman et al. 的 `target leakage` 更接近“本应由方法发现的 target information 被输入”；Deng et al. 的 `evaluation contamination` 对应 benchmark answer 已进入生成过程；Yang et al. 则直接证明 agent-computer interface 会改变 agent performance。这里最实用的检索词是 `target leakage in evaluation`、`benchmark contamination`、`agent scaffold ablation`、`scaffold confounding` 和 `capability attribution to scaffolds`。

**Falsifiable prediction.** 去掉 shared prompt 中的 closing-speed formula 和 one-off-event skeleton 后，gentleness compliance 不会再是 **16/16**，候选 reward 的结构方差会上升；若 CREATE 的 trajectory evidence 真有独立作用，它与 EUREKA 的差距应只在 interface-only cell 出现，而在 oracle-scaffold cell 缩小。

**Defeat condition.** 若 preregistered factorial experiment 显示有无 skeleton 时 CREATE–EUREKA gap 基本相同，或者 interface-only 条件下两者仍同样产生、同样训练失败，那么“shared scaffold 是没有方法差异的主要原因”即被否定。

## P4 — What the gate can and cannot establish

任何有限、固定的 probe library 都存在一个简单的不识别性问题：两个 reward 可以在 library 上完全相同，却在未覆盖状态上不同；即使它们对全部相关 trajectories 的 ordering 一致，PPO 的 gradient scale、reward variance、credit horizon 与 exploration landscape 仍可能不同。因此，training-free gate 无法同时证明 objective correctness 和 learnability。当前 gate 不是“样本还少但看起来不错”，而是只有 negative evidence：**correctly rejected two 0/60 candidates**，同时 **accepted three 0/60 candidates**。

scripted-controller returns 仍值得做，但应作为 rejection test。使用真实 wrapper、clipping、discount 与 termination 计算完整 trajectory return，可以发现局部 check 看不到的 component competition。那个以 **1.555 m/s** 入 dock 的候选可能在“同 contact、两种 speed”的局部比较上方向正确，却在完整 approach trajectory 上仍把快速 progress 排得更高。没有 boundary guard 的候选则应被 candidate-return-maximizing controller 引向 out-of-bounds，从而被拒绝。

reference trajectory library 也应由 matched pairs 构成，而不是简单分成 good/bad examples；否则长度、初始条件或位置差异会成为混杂因素。用开发集设计 pairs，用 controller-family-held-out 集验证，且不把具体 pairs 或正确答案写进 prompt。进一步可按 Bıyık and Sadigh 的 active preference learning 思路，从大量 reachable pairs 中选择“候选 reward ordering 分歧最大”的 pair，再由 task oracle 标注；Reddy et al. 的 ReQueST 则给出了主动寻找高预测 reward、低真实 utility 行为以暴露 reward hacking 的直接先例。

一个 check 要有 discriminative power，至少必须满足两点：在当前 candidate distribution 上有非零输出方差，并且该方差与 held-out training outcome 有稳定关联。v4 gentleness check 因所有候选复制同一 snippet 而成为 **16/16**，所以第一项已经失败；无论它在语义上多合理，其 mutual information 都是零。一般构造方法是先生成候选，再在未泄露给生成器的 reachable trajectory pool 中寻找 candidate disagreement 最大的 tests，最后保留独立 held-out controller families 估计关联；不能先把 check 的实现写进 prompt，再把 compliance 当成 quality evidence。

正向 selection 应采用 multi-fidelity training。具体建议是先让 structural gate 仅拒绝明显违反 critical pair、runtime、boundary 或 completion-credit 条件的候选；余下候选各训练 `0.1M–0.3M` steps，即 3M full budget 的约 **3–10%**。第一 rung 不按 generated return 排名，而按未复用 seeds 上的 maximum settled streak、dock-entry speed、below-`0.05` dock occupancy、out-of-bounds rate 和 hard-impact rate 排序；只提升前一部分到 `0.6M`，再提升少数到 full budget。这属于 successive halving/Hyperband，而不是另一个静态 gate。最终 fresh-60 必须永久 hold out。

要声称 gate 具有 positive predictive validity，必须在从未参与 probe 设计、也没有共享 snippet 的候选上 preregister threshold，并报告 fresh-success 的 precision、recall、AUPRC 与置信区间；“约 1000x dynamic range”本身没有预测意义，因为单调变换可以任意扩大动态范围而不改变排序。

**Falsifiable prediction.** 扩展 trajectory gate 会拒绝已知的 out-of-bounds 与 **1.555 m/s** 候选，但在控制这些 failure modes 后，gate-pass 本身仍不能可靠预测 fresh-60 success；加入 `0.1M–0.3M` multi-fidelity behavior rung 后，候选排序与 full-budget outcome 的关联才会明显上升。

**Defeat condition.** 如果在至少 30 个全新、独立生成并完成 full-budget training 的候选上，预注册的 training-free gate 达到 Spearman correlation `>0.5`，且 successful candidate 的 PPV `>0.5`、其 95% confidence lower bound 仍高于 candidate-pool base rate，那么我关于“它不能作为正向 selector”的判断是错的。

# Adversarial cross-examination (second reply)

我认为最可能错误的 claim 是 Section 4, P2 中的：“把 closing-speed formula 和 one-off-event skeleton 放进 shared prompt，是该环境目前没有 CREATE-versus-EUREKA difference 的 main reason。”

现有数据证明了 skeleton 会造成 structural convergence：gentleness check 为 **16/16**，all-four pass 为 **15/16**。但它没有证明这种 convergence 是 performance gap 消失的主要原因。事实上，三个 passers 仍然都是 **0/60**，说明共享代码只统一了表面结构，没有统一正确 weighting、boundary coverage 或 learnability。另一种同样符合数据的解释是：该任务从 naive initial reward 的 **≈2%** threshold 区域起步，两个方法都被更基础的 exploration/credit-assignment failure 压在 floor 上；即使没有 shared skeleton，也可能仍然没有可测 gap。

证明我错的方法很直接：做 `method × scaffold` factorial experiment，固定 model、generation count、training budget 和 seed protocol。如果 interface-only 条件下 CREATE 稳定优于 EUREKA，而加入共享 skeleton 后差距消失，并且这种 interaction 可由 copied mechanism 的比例解释，那么“shared skeleton 是 main reason”得到支持。没有这个 interaction，目前最多只能说它是一个强 confound，不能说是主因。

# Confidence self-assessment (second reply)

P1：**4/5**。论证较强；丢失的 1 分来自 literature gap：reward comparison、IRL feature expectations、hypothetical queries 各自成熟，但“把这些字段作为 LLM reward-repair evidence”是新的组合，没有直接论文验证。

P2：**3/5**。operational criterion 可执行，反例也稳定；低分主要来自 insufficient literature，因为 `target leakage` 与 `scaffold confounding` 是相邻概念，尚无公认论文给出 reward-search prompt 中 interface/solution 的精确边界。

P4：**4/5**。有限 probe 不能保证 optimization learnability 的论证较强，multi-fidelity selection 也有成熟依据；剩余不确定性来自当前 candidate pool 太小，尚不能排除一个针对该固定环境高度预测性的 trajectory gate。

如果只能做一件事：本周生成一个不向 LLM 公开的 reachable matched-trajectory library，并用它对所有历史 rewards 计算 full-trajectory pairwise ranking，再检验该分数是否能在不含 shared snippet 的候选上区分 `0/60` 与非零 fresh-60。
