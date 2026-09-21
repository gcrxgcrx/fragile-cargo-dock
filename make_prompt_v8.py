"""Derive the v8 prompt from v7 by four auditable replacements.

Why v8 exists (all four reasons are measurements, see
`runs/env_007/V8_PROMPT_PREREGISTRATION.md`)

1. **v7 contradicts itself.** One clause requires a per-step payoff on the settled state
   (self-check ④: calling the reward 12 times on a settled state must grow linearly), while
   the `完成状态下…其余组件必须恰好为 0` clause requires every component except the one-off
   event to contribute **exactly 0** when the completion predicate holds — the same state.
   All eight v7 candidates resolved it the same way: they gate the per-step payoff on
   "the one-off event has not fired yet", or zero it once it fires.

2. **The one-off event is worth exactly nothing under the harness clip.** Measured on a real
   success trajectory with clip 20 (`probe_v7_shape.py`): the terminal step's raw
   `+300 (event) + 20 (stream) = 320` is clipped to `+20`, which is bit-for-bit what a
   single settled step pays. So `10 * B > 3 * (process bound * 400)` — the rule v5 added and
   v7 kept — spends the model's effort on a term the harness deletes.

3. **The `rest must be 0` rule re-created the procrastination bug it was meant to prevent.**
   Because the stream switches off when the event fires (or after it), *not* completing
   keeps paying while completing stops the episode. Measured for `cand_00`: 876.7 (truncated
   + 40 settled steps) vs 96.7 (the actual successful episode).

4. **The measured cure for grinding is boundedness, not a ban.** The working hand-written
   arms (`probeD` 73.3 %, `probeD`/`probeA`) pay a constant `+20` per settled step — an
   unbounded state reward — and they work; what distinguishes the arm that ground in the
   dock for 43 steps is that its *progress* term was a closed-loop integral (`max(0, Δd)`
   with no penalty for moving away), which a hand-written control with a true potential
   difference never has.

What v8 changes, and nothing else:
  (i)   the one-off event becomes **optional**, with the clipping arithmetic stated;
  (ii)  the settled-state per-step payoff must **not be switched off** by the event;
  (iii) the "everything else must be exactly 0" rule is **narrowed** to persistent *state*
        bonuses, with the per-step success-predicate payoff explicitly exempt;
  (iv)  the progress term must be a **signed potential difference** (a cycle must not pay).

Usage:
    python make_prompt_v8.py
"""

from __future__ import annotations

import difflib
from pathlib import Path

SRC = Path("prompts/eureka_01_initial_reward_v7.md")
DST = Path("prompts/eureka_01_initial_reward_v8.md")
DIFF = Path("runs/env_007/v8_prompt.diff")

# --- (i)+(ii) the requirement list -----------------------------------------
REQ_OLD_V7 = """- 完成奖励必须**一次性**；同一 episode 内不得重复发放。
- **同时**必须有一项"停稳期每步收益"：当"泊位内 + 对齐 + 慢"成立时每步给正收益（参考量级 +20/步）。
  它不是可选加分项，而是必要条件。
- 若使用模块级状态，**必须**用 `obs[18]` 检测回合边界并重置，否则状态会跨 episode 污染。
- 一次性的"首次进入泊位"奖励可以保留（整局只发一次），但**它不能替代停稳期每步收益**：
  实测中只带一次性进入奖励的候选，入坞率能到 0.97，交付率仍然 **0/60**。
"""

REQ_NEW_V8 = """- **同时**必须有一项"停稳期每步收益"：当"泊位内 + 对齐 + 慢"成立时每步给正收益（参考量级 +20/步）。
  它不是可选加分项，而是必要条件。这是本版本唯一必须存在的"完成侧"信号。
- **这一项不得被你自己的其它逻辑关掉**：不得因为"一次性完成事件已经发放过"、
  因为某个连续计数已经达标，或任何其它条件而停止发放。谓词成立就每步发放。
- 一次性完成事件**可以保留、也可以完全不写**（见下面那条关于单步裁剪的说明，它在本环境里
  并不承担实际作用）。若保留，同一 episode 内不得重复发放。
- **进阶自检（新增）**：你必须保证"推进项"是**有符号**的：
  `本帧更接近则给正分，本帧更远则给负分（对称），按米计`。
  **禁止**只奖励接近、不惩罚远离（例如 `max(0, 距离差)`），也禁止让同一段路反复计分——
  实测中这样的写法会让同一个候选在单回合里累积到 **2747** 分，而几何上界只有约 **450**，
  即策略在泊位附近来回刷分：该候选最终只交付 **3/60**。
- 若使用模块级状态，**必须**用 `obs[18]` 检测回合边界并重置，否则状态会跨 episode 污染。
- 一次性的"首次进入泊位"奖励可以保留（整局只发一次），但**它不能替代停稳期每步收益**：
  实测中只带一次性进入奖励的候选，入坞率能到 0.97，交付率仍然 **0/60**。
"""

# --- (iii) the satisfaction/zeroing rule ------------------------------------
ZERO_OLD_V7 = """# 完成状态下，除一次性事件外其余组件必须恰好为 0（违反即无效）

上一条只要求"一次性事件"，做到还不够。实测中大量候选写出了正确的一次性事件，却仍然失败，
原因是它们在泊位内**顺手**留了一个很小的持久项，例如：

```python
align_bonus = 0.3 * (货箱在容差内且对齐)     # 看起来无害
```

单步 0.3 看似微不足道，但 episode 有 400 步，累积是 **120**；而 +300 的一次性事件会被单步裁剪
压到 **+20**。**120 > 20，持久项照样赢。** 实测这样的候选交付率仍然是 0。

要求：

- 当货箱处于完成状态（在容差内 + 对齐 + 慢）时，**除一次性完成事件之外，其他所有组件的
  单步贡献之和必须恰好为 0**。
- 不允许任何"靠近目标就持续给分"的状态项——对齐奖励、稳定奖励、低速奖励、接近度奖励都一样。
  这类量只能作为**门控**乘在**增量**信号上（例如 `对齐度 × 本帧的靠近增量`），或只作为**惩罚**
  （例如速度过快时扣分），**不得单独作为正项按步累积**。
- 唯一的例外是"首次进入泊位"这类**一次性**事件——它们整局只发一次，不随步数累积，可以保留。

**必做自检 ⑤**：把 reward 在**同一个"停稳"状态**上连续调用 12 次。
要求：**除了发一次性事件的那一次之外，其余 11 次的返回值必须恰好为 0**（|值| < 1e-6）。
只要有任何一次明显非零（哪怕是 0.3），就说明存在持久项，必须删除或改成门控/增量形式。
"""

ZERO_NEW_V8 = """# 禁止"顺手留下"的持久**状态**奖励（违反即无效）

实测中大量候选写对了完成结构，却仍然失败，原因是它们在泊位内**顺手**留了一个很小的持久项，例如：

```python
align_bonus = 0.3 * (货箱在容差内且对齐)     # 看起来无害
```

单步 0.3 看似微不足道，但 episode 有 400 步，累积是 **120**；而 +300 的一次性事件会被单步裁剪
压到 **+20**。**120 > 20，持久项照样赢。** 实测这样的候选交付率仍然是 0。

要求：

- 不允许任何"靠近目标就持续给分"的**状态**项——对齐奖励、稳定奖励、低速奖励、接近度奖励都一样。
  这类量只能作为**门控**乘在**增量**信号上（例如 `对齐度 × 本帧的靠近增量`），或只作为**惩罚**
  （例如速度过快时扣分），**不得单独作为正项按步累积**。
- 唯一的例外是"首次进入泊位"这类**一次性**事件（整局只发一次，不随步数累积）。
- **本版本明确豁免**：前面要求的"停稳期每步收益"**不属于**被禁止的持久状态项——它是本题的
  必要完成信号，谓词成立时必须每步发放，**不得**被一次性事件或任何计数清零。

**必做自检 ⑤（本版本已改写）**：把 reward 在**同一个"停稳"状态**上连续调用 12 次。
要求：**每次的差值都等于同一个正数**（即停稳收益每步都在发），若某次之后突然变成 0，
说明你的停稳收益被自己的完成事件关掉了，必须去掉那个开关。
"""

# --- (i) the template that forces a +300 event ------------------------------
TPL_OLD_V7 = """success_event = 0.0
if _STREAK[0] >= 10 and not _PAID[0]:
    _PAID[0] = True
    success_event = 300.0                   # 一次性，整个 episode 只发一次
```"""

TPL_NEW_V8 = """success_event = 0.0
if _STREAK[0] >= 10 and not _PAID[0]:
    _PAID[0] = True
    success_event = 300.0                   # 一次性，整个 episode 只发一次
```

**关于一次性事件的量级（实测，请按此调整你的设计权重）：** 本环境对**单步**奖励做裁剪，
上限为 **20**。因此在发放一次性事件的那一步，`事件 + 停稳收益` 会被整体压到 20——
与一个普通停稳步**完全等值**。也就是说：**一次性事件在本环境里不承担实际作用**，
真正让策略"进入并保持停稳"的是那条每步收益。所以：

- **不要**为了满足某个"完成事件必须压过全程过程收益"的量级公式而把一次性事件写得很大
  （那正是上一版提示词要求 `10 * B > 3 * (过程项上限 × 400)` 的后果，实测无效）；
- 正相反：如果你把**过程项**（尤其是推进项）写得很大，它会把每步收益的效果稀释掉——
  实测中推进项过强的候选会"到得了泊位、却完不成 10 步保持"（入坞率 0.95、交付 0/60）。
- 结论：把**每步**信号的量级做对，一次性事件可写可不写。"""


def main() -> None:
    src_text = SRC.read_text(encoding="utf-8")
    text = src_text
    applied = []

    for name, old, new in [
        ("requirement list (+ cycle-neutral progress, + do not switch the payoff off)",
         REQ_OLD_V7, REQ_NEW_V8),
        ("completion-state zeroing -> narrowed to state bonuses, payoff exempt",
         ZERO_OLD_V7, ZERO_NEW_V8),
        ("one-off template gains the clipping arithmetic", TPL_OLD_V7, TPL_NEW_V8),
    ]:
        if old not in text:
            raise SystemExit(f"anchor not found for '{name}'; refusing to patch blind")
        text = text.replace(old, new, 1)
        applied.append(name)

    header = ("<!-- v8 = v7 with three auditable replacements, generated by\n"
              "     make_prompt_v8.py. Each change is backed by a measurement; the rationale,\n"
              "     the protocols and the predictions are in\n"
              "     runs/env_007/V8_PROMPT_PREREGISTRATION.md. v7 is left untouched. -->\n\n")

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
