"""Derive v9 from v8 by adding the *structure* of the environment's own reward.

Rationale (see `runs/env_007/V9_STRUCTURE_PREREGISTRATION.md`)

The measured situation when v9 is written:

* the environment's own reward, fed through the wrapper at 1.2 M / clip 20, scores
  **47 / 52 / 54 / 56 out of 60 across four training seeds** (`passthrough_probe/seed*`),
  i.e. it is *stable* (sd ~ 4), while every hand-written and LLM-written reward tried so far
  is both weaker and far more seed-dependent (`control_v1`: 0 / 35 / 43 / 54; LLM v8
  candidate: 4/60);
* of the nine native reward terms, **only the impulse-based one** (`roughness`, and the
  `hard_hit` count that shares its signal) needs a quantity that the contract forbids
  (`peak_impulse`, available only in `info`). The other seven are exactly recoverable from
  `obs` and `action`;
* a code-level re-expression of the native reward reaches 98.3 % (`probeC`), and the same
  thing with an observation-only proxy for the impulse term reaches 73.3 % (`probeD`).

So the information needed to write a *working* reward is known to be available to a
generator that reads the environment — but the generator has never been given it. v9 gives it,
while keeping every privileged channel (the `impulse` quantity and the `info` dictionary)
closed, which is the contract this study is about.

Six auditable replacements against v8; nothing else changes.

Usage:
    python make_prompt_v9.py
"""

from __future__ import annotations

import difflib
from pathlib import Path

SRC = Path("prompts/eureka_01_initial_reward_v8.md")
DST = Path("prompts/eureka_01_initial_reward_v9.md")
DIFF = Path("runs/env_007/v9_prompt.diff")

# ---------------------------------------------------------------------------
# 1. the structure block, inserted at the top of the design method
# ---------------------------------------------------------------------------
METHOD_ANCHOR = """# 设计方法

1. 先确定任务的主行为语义（例如：任务进展、状态稳定、目标事件、动作代价）。
2. 把每个语义实现为一个**具名组件**，各自独立记录在 `components` 字典中。
3. 总奖励是各组件之和，权重决定其量级。
4. 组件要和任务成败同向：能让策略真正完成任务，而不是只优化代理指标。
5. 注意量级平衡：单个惩罚项每步的绝对值不应压过主信号；过多的惩罚会让策略不敢动。
"""

METHOD_V9 = """# 已知的奖励结构（本版本新增，**必须照此实现**）

本环境的作者奖励函数是已知的，其**分项语义与权重**如下。你的任务是：**用 `obs` 与 `action` 把它重写出来。**
（注意：这些是"设计结构"，不是"可以直接读取的字段"——你**不能**读 `info`，也拿不到任何未在观测中声明的量。）

| 分项 | 权重 | 语义 |
|---|---:|---|
| `approach_cargo` | **+1.0 / 米** | 本帧小车→货箱距离缩短了多少米（势函数形式） |
| `progress` | **+1.0 / 米** | 本帧货箱→泊位距离缩短了多少米（势函数形式） |
| `dock_enter` | **+5.0** | 货箱**首次**完全进入泊位容差时，一次性发放 |
| `roughness` | **−0.02 / (N·s)** | 接触冲量比例惩罚：撞得越狠罚越多（"轻拿轻放"） |
| `action_cost` | **−0.0005** | 动作平方和 |
| `time_cost` | **−0.002 / 步** | 每步固定时间成本 |
| `hard_hit` | **−0.5** | 单步内发生一次硬冲击的固定惩罚 |
| `terminal_success` | **+300** | 完成条件**连续 10 步**成立时发放一次 |
| `terminal_failure` | **−100** | 货箱或小车越界、或累计硬冲击 ≥3 次时发放一次 |

把这些分项重写成 `obs` + `action` 的表达式时，逐项的可行性与做法如下：

1. **`approach_cargo`**：小车→货箱距离（由 `obs[6]`, `obs[7]` 与姿态还原）**本帧的差**，按米计。
2. **`progress`**：货箱→泊位距离（由 `obs[12]`, `obs[13]` 还原）**本帧的差**，按米计。
   **两项都必须是有符号、对称的**：靠近给正分，远离给负分。
3. **`dock_enter`**：用模块级状态记录"是否曾经完全进入容差"，只发一次。
4. **`action_cost` / `time_cost`**：直接照写。
5. **`terminal_success`**：用模块级状态对"泊位内 + 对齐 + 慢"做**连续计数**，达到 10 次时发一次。
   注意本环境在满足时会**立即终止** episode，所以它天然最多只发一次。
6. **`terminal_failure`**：越界由 `obs[0]`/`obs[1]` 判断（`|x|` 或 `|y|` 超过约 1.05 即出界）。
7. **`roughness`（唯一需要代理的项）**：观测里**没有冲量**这一维。请构造一个可观测的
   "接近速度 × 接触"代理——例如
   `closing = obs[4]*3.0 - (obs[8]*3.0*obs[2] + obs[9]*3.0*obs[3])`，再取 `max(0, closing)`，
   仅在实际接触（`obs[14] > 0.5`）时乘以一个负系数。**注意两点**：
   该项是用来**抑制撞击**的，不是用来压制推进的——实测把它的量级做到与推进项相当，
   会让策略连动都不敢动（入坞率 0.00）。**它必须比推进项弱**，只在"撞得很快"时显著。

**优先级声明**：本节给出的分项表**优先于**下文任何与之冲突的通用规则；若下文的规则与本节冲突，以本节为准。

# 设计方法

1. 先确定任务的主行为语义（例如：任务进展、状态稳定、目标事件、动作代价）。
2. 把每个语义实现为一个**具名组件**，各自独立记录在 `components` 字典中。
3. 总奖励是各组件之和，权重决定其量级。
4. 组件要和任务成败同向：能让策略真正完成任务，而不是只优化代理指标。
5. 注意量级平衡：单个惩罚项每步的绝对值不应压过主信号；过多的惩罚会让策略不敢动。
"""

# ---------------------------------------------------------------------------
# 2-5. the four v8 rules that the v9 recipe supersedes, made explicit
# ---------------------------------------------------------------------------
CYCLE_OLD = """- **进阶自检（新增）**：你必须保证"推进项"是**有符号**的：
  `本帧更接近则给正分，本帧更远则给负分（对称），按米计`。
  **禁止**只奖励接近、不惩罚远离（例如 `max(0, 距离差)`），也禁止让同一段路反复计分——
  实测中这样的写法会让同一个候选在单回合里累积到 **2747** 分，而几何上界只有约 **450**，
  即策略在泊位附近来回刷分：该候选最终只交付 **3/60**。
"""
CYCLE_NEW = """- **推进项必须是有符号、对称的**（与上面分项表第 2 条一致）：
  `本帧更接近给正分，本帧更远给负分`，按米计；**禁止** `max(0, 距离差)` 这种只奖不罚的写法
  （实测会让单回合累积到 2747 分，而几何上界约 450，策略在泊位附近来回刷分，最终 3/60）。
"""

ONE_OFF_OLD = """**关于一次性事件的量级（实测，请按此调整你的设计权重）：** 本环境对**单步**奖励做裁剪，
上限为 **20**。因此在发放一次性事件的那一步，`事件 + 停稳收益` 会被整体压到 20——
与一个普通停稳步**完全等值**。也就是说：**一次性事件在本环境里不承担实际作用**，
真正让策略"进入并保持停稳"的是那条每步收益。所以：

- **不要**为了满足某个"完成事件必须压过全程过程收益"的量级公式而把一次性事件写得很大
  （那正是上一版提示词要求 `10 * B > 3 * (过程项上限 × 400)` 的后果，实测无效）；
- 正相反：如果你把**过程项**（尤其是推进项）写得很大，它会把每步收益的效果稀释掉——
  实测中推进项过强的候选会"到得了泊位、却完不成 10 步保持"（入坞率 0.95、交付 0/60）。
- 结论：把**每步**信号的量级做对，一次性事件可写可不写。
"""
ONE_OFF_NEW = """**关于量级（实测，请按此调整）：** 本环境对**单步**奖励做裁剪，上限为 **20**。
因此：**一次性事件（+300）在第 10 个停稳步上会被裁到 20**，与一个普通停稳步等值——
它不承担实际作用，真正让策略"进入并保持停稳"的是每步收益（按上面的分项表实现即可）。
但**不要**为了"让事件主导"而把过程项写得很大：实测推进项过强的候选会"到得了泊位、
却完不成 10 步保持"（入坞率 0.95、交付 0/60）。
"""

NO_STATE_OLD = """- **本版本明确豁免**：前面要求的"停稳期每步收益"**不属于**被禁止的持久状态项——它是本题的
  必要完成信号，谓词成立时必须每步发放，**不得**被一次性事件或任何计数清零。
"""
NO_STATE_NEW = """- **豁免**：上面分项表里的 `terminal_success`（连续 10 步谓词）与 `dock_enter`
  都是完成侧信号，**不属于**被禁止的"顺手留下的持久状态项"。但它们也不必每步发放——
  按分项表实现即可（`terminal_success` 是一次性事件）。
"""

# ---------------------------------------------------------------------------
# 6. the requirement list gains the re-expression requirement
# ---------------------------------------------------------------------------
REQ_ANCHOR = """- **同时**必须有一项"停稳期每步收益"：当"泊位内 + 对齐 + 慢"成立时每步给正收益（参考量级 +20/步）。
  它不是可选加分项，而是必要条件。这是本版本唯一必须存在的"完成侧"信号。
"""
REQ_NEW = """- 必须实现**上面分项表里的每一项**（用 `obs`+`action` 重写），并为 `roughness` 构造观测代理。
- 完成侧信号按分项表实现（`dock_enter` +5 一次性、`terminal_success` 连续 10 步一次性 +300）。
"""


def main() -> None:
    src_text = SRC.read_text(encoding="utf-8")
    text = src_text
    applied = []

    for name, old, new in [
        ("结构配方块（分项语义与权重 + 7 项重写指引 + 优先级声明）", METHOD_ANCHOR, METHOD_V9),
        ("推进项规则与配方对齐", CYCLE_OLD, CYCLE_NEW),
        ("一次性事件说明与配方对齐", ONE_OFF_OLD, ONE_OFF_NEW),
        ("状态项豁免与配方对齐", NO_STATE_OLD, NO_STATE_NEW),
        ("要求清单改为&#39;照配方实现每一项&#39;", REQ_ANCHOR, REQ_NEW),
    ]:
        if old not in text:
            raise SystemExit(f"anchor not found for '{name}'; refusing to patch blind")
        text = text.replace(old, new, 1)
        applied.append(name)

    header = ("<!-- v9 = v8 + the known reward structure (per-term semantics and weights) with every\n"
              "     privileged channel still closed. Generated by make_prompt_v9.py; rationale,\n"
              "     protocol and predictions in runs/env_007/V9_STRUCTURE_PREREGISTRATION.md.\n"
              "     v8 is left untouched. -->\n\n")

    DST.write_text(header + text, encoding="utf-8")
    diff = difflib.unified_diff(src_text.splitlines(), text.splitlines(),
                                fromfile=str(SRC), tofile=str(DST), lineterm="", n=2)
    DIFF.write_text("\n".join(diff) + "\n", encoding="utf-8")
    print(f"applied {len(applied)} replacements:")
    for a in applied:
        print(f"  - {a}")
    print(f"\nwrote {DST} ({len(text)} chars) and {DIFF}")


if __name__ == "__main__":
    main()
