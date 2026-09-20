"""Derive the v7 prompt from v5 by three auditable rule replacements.

Why v7 exists
-------------
The repair study (`runs/env_007/REPAIR_TEST_FINDINGS.md`) measured what a working reward for
this task needs, and the current scaffold contradicts one of the two conditions:

* **v5 forbids the ingredient.** v5 lines 112–119: "完成奖励必须是一次性事件，不能在'成功状态'上每步给分
  （违反即无效）", with self-check ④ ("把 reward 在同一个'停稳'状态上连续调用 12 次 … 若线性增长，
  说明你写成了每步流，必须重写") enforcing it. But the measured two-edit repair that takes two
  independent candidates from 0/60 to 23/60 is (1) the candidate's own approach coefficient ×50
  and (2) **a dense +20/step payoff on the instant the success predicate holds** — exactly the
  form v5 bans.
* **The ban rests on a mis-attributed observation.** v5's own justification cites a measured
  failure: a per-step settled payoff made a policy grind in the dock for 43 steps without ever
  completing 10 consecutive ones (0 % delivery). That measurement is real — it is `control_v1`
  (`SESSION_STATE.md` §3b: progress + approach + a `+20`/step settled stream, 0/60). But the
  grinding was not caused by the stream; it was caused by the *approach* term being far too
  weak, so re-farming the stream beat completing. Measured: `control_v1` has `A = −0.26` (acting
  worse than idling). Add the ×50 approach term to the reward that has the stream and delivery
  appears (`r09`: 23/60, dock 0.97); the stream alone does nothing (`r06`, `g01_stream`: metrics
  bit-for-bit identical to their copy controls).
* **The prescribed cure does not work either.** v5 line 154 offers a one-off "first dock entry"
  bonus as the fix for "gets in but cannot stop". Measured: arm `r04` keeps that one-off entry
  bonus and reaches `dock_entered` 0.97 with **0/60** — it still cannot stop.
* **A measured caution about gentleness.** v5 mandates a gentleness signal. Adding
  `probeD`'s gentleness on top of a candidate that already penalises closing speed *destroyed*
  the behaviour (`r08`: 0/60 with dock entry 0.00; `r10`: 0/60). So v7 keeps the requirement but
  forbids stacking duplicates that swamp the completion payoff.

The three replacements below are applied to a copy of v5; the script refuses to patch if any
anchor is missing, and writes the resulting prompt plus a unified diff for the record.

Usage:
    python make_prompt_v7.py
"""

from __future__ import annotations

import difflib
from pathlib import Path

SRC = Path("prompts/eureka_01_initial_reward_v5.md")
DST = Path("prompts/eureka_01_initial_reward_v7.md")

# --- 1. the ban -> a conditional rule -------------------------------------
BAN_OLD = """# 完成奖励必须是一次性事件，不能在"成功状态"上每步给分（违反即无效）

环境在"连续 10 步满足完成条件"时会**立即终止 episode**。由此：

- 如果你在"货箱在泊位内 + 对齐 + 慢"这个**状态**上**每步**给分，那么策略的最优解是
  **永远不要走完第 10 步**——只要在第 9 步前后被蹭一下、让连续计数清零，这条收益流就能一直续着。
  实测这种写法让策略在泊位里磨 43 步，30/60 回合达到 5–9 步却从未连续 10 步，交付率 0%。
- 正确写法是**一次性事件**：整个 episode 只发一次。
"""

BAN_NEW = """# 完成奖励：一次性事件 **和** 停稳期的每步收益 **都要有**（两者缺一不可）

环境在"连续 10 步满足完成条件"时会**立即终止 episode**。历史上有一种写法被禁掉了，理由是
"在成功状态上每步给分会让策略磨在泊位里不走完第 10 步"（实测：磨 43 步、30/60 回合到 5–9 步、
交付率 0%）。**那条禁令是错的**：把同一条每步收益加上一个足够强的推进项之后，
交付率从 0% 变成 38%（同一环境、同预算、同一评测）。真正的原因不是"每步给分"，
而是**推进项太弱**，以至于反复续着停稳收益比走完 10 步更划算。

因此本版本要求你**同时**写出：

1. **一次性完成事件**（整个 episode 只发一次）：策略有理由**走完**最后 10 步；
2. **停稳期的每步收益**（在"泊位内 + 对齐 + 慢"成立时，每步给一个正收益）：策略有理由**进入并保持**
   这个状态。没有它，实测的候选会在泊位外停住、永远不满足完成谓词。

两者共存时，唯一的风险是"续着停稳收益而不走完 10 步"。判定标准是**推进项与停稳项的强度比**：

- 走完 10 步应当比"赖在停稳状态"更划算。做法是让**推进项**足够强（见自检⑤的量级要求），
  而不是取消停稳收益。
- 若你的奖励里推进项很弱、停稳项却很密，那就会复现 0% 的磨蹭行为；这时要**加强推进项**，
  不要删掉停稳收益。
"""

# --- 2. the requirement list ----------------------------------------------
REQ_OLD = """- 完成奖励必须**一次性**；同一 episode 内不得重复发放。
- 若使用模块级状态，**必须**用 `obs[18]` 检测回合边界并重置，否则状态会跨 episode 污染。
- 可以额外给一次性的"首次进入泊位"奖励（同样整局只发一次），它能帮助策略跨过"进得去但停不住"的坎。
"""

REQ_NEW = """- 完成奖励必须**一次性**；同一 episode 内不得重复发放。
- **同时**必须有一项"停稳期每步收益"：当"泊位内 + 对齐 + 慢"成立时每步给正收益（参考量级 +20/步）。
  它不是可选加分项，而是必要条件。
- 若使用模块级状态，**必须**用 `obs[18]` 检测回合边界并重置，否则状态会跨 episode 污染。
- 一次性的"首次进入泊位"奖励可以保留（整局只发一次），但**它不能替代停稳期每步收益**：
  实测中只带一次性进入奖励的候选，入坞率能到 0.97，交付率仍然 **0/60**。
"""

# --- 3. self-check 4 -> the advantage-scale check --------------------------
CHECK_OLD = """**必做自检 ④**：把 reward 在**同一个"停稳"状态**上连续调用 12 次。
要求累计值**在第一次之后就停止增长**；若线性增长，说明你写成了每步流，必须重写。
"""

CHECK_NEW = """**必做自检 ④**：把 reward 在**同一个"停稳"状态**上连续调用 12 次。
要求累计值**线性增长**（每步都有停稳收益）；若第一次之后就不再增长，说明你漏掉了停稳期每步收益，
必须补上。

**必做自检 ⑤（本版本新增，最重要）**：估算三条轨迹的**每步平均奖励**，并检查单调性与量级：

| 行为 | 每步平均奖励 |
|---|---|
| 什么都不做（idle） | R_idle |
| 朝泊位推货箱（正常推进） | R_push |
| 停稳在泊位内（满足完成谓词） | R_settled |

必须同时满足：

1. `R_push > R_idle` **且差距明显**：`R_push − R_idle` 不得小于你所有**罚项**在单步上的最大量级。
   实测中大量候选失败的原因就是"行动比不动还差"（推进项被罚项淹没），于是策略学会**什么都不做**——
   奖励的最优点变成了静止。
2. `R_settled > R_push`：停稳必须是全局最优点，而不只是"不被罚"。
3. 量级要求：`R_push − R_idle` 至少要与你最大的推进/停稳项同量级，**不能是它的千分之一**。
   如果你为了让奖励"温和"而把所有正项都压得很小，PPO 学不到东西。

把这三行数字写进注释里，作为你的自检记录。
"""

# --- 4. gentleness: keep, but forbid swamping duplicates -------------------
GENTLE_OLD = "# 必须包含\"接触轻柔度\"信号（违反即无效）\n"
GENTLE_NEW = ("# 必须包含\"接触轻柔度\"信号（或等价的既有信号），但不得堆叠到淹没完成收益\n"
              "> 若你的奖励里**已经**有惩罚接近速度/高速接触的项，不要再用同一个公式叠加一个：\n"
              "> 实测中把轻柔项叠在已有速度罚之上，会让策略连泊位都不进（入坞率 0.00、交付率 0/60）。\n"
              "> 轻柔信号的作用是**抑制撞击**，不是把推进项压到没有梯度。\n")


def main() -> None:
    src_text = SRC.read_text(encoding="utf-8")
    text = src_text
    applied = []

    for name, old, new in [
        ("ban -> conditional rule", BAN_OLD, BAN_NEW),
        ("requirement list", REQ_OLD, REQ_NEW),
        ("self-check 4 -> 4+5", CHECK_OLD, CHECK_NEW),
        ("gentleness clause", GENTLE_OLD, GENTLE_NEW),
    ]:
        if old not in text:
            raise SystemExit(f"anchor not found for '{name}'; refusing to patch blind")
        text = text.replace(old, new, 1)
        applied.append(name)

    header = ("<!-- v7 = v5 with four auditable rule replacements, generated by\n"
              "     make_prompt_v7.py. Rationale and the measurements behind each change:\n"
              "     runs/env_007/V7_PROMPT_PREREGISTRATION.md. v5 is left untouched. -->\n\n")
    DST.write_text(header + text, encoding="utf-8")

    diff = difflib.unified_diff(src_text.splitlines(), text.splitlines(),
                                fromfile=str(SRC), tofile=str(DST), lineterm="", n=2)
    Path("runs/env_007/v7_prompt.diff").write_text("\n".join(diff) + "\n", encoding="utf-8")
    print(f"applied {len(applied)} replacements: {', '.join(applied)}")
    print(f"wrote {DST} ({len(text)} chars) and runs/env_007/v7_prompt.diff")


if __name__ == "__main__":
    main()
