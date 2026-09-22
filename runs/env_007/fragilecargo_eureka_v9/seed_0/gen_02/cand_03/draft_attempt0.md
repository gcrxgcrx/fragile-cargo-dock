训练反馈显示：settle_bonus(207) + terminal_success(161) 占了总回报 ~92%，而任务分数只有 3.9、20/20 全是 truncation——说明智能体学会了用「离 dock 中心 0.25m 的宽松代理 + 二值门控」刷分，但从未真正满足环境的完整停靠条件（完全入坞/对齐/停稳）。progress/approach 等真实进展信号被完全压制。因此我把停靠奖励从「二值门控」改成「以 dock 接近度×对齐×低速构成的连续门控函数」（靠近中心才有梯度、且平滑引导更精确入坞），降低其量级与 one-shot 权重，提高运输进展权重，并收紧完成谓词，避免宽松代理被收割。

```python
_PREV_T = [-1.0]
_STREAK = [0]
_PAID_SUCCESS = [False]
_ENTERED = [False]
_HARD_HITS = [0]
_PAID_FAIL = [False]


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 回合边界检测（obs[18] 单调递增，重置时回落）----
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 0.0025:
        _STREAK[0] = 0
        _PAID_SUCCESS[0] = False
        _ENTERED[0] = False
        _HARD_HITS[0] = 0
        _PAID_FAIL[0] = False
    _PREV_T[0] = t

    # ---- 由 obs 还原几何量（米）----
    cart_x = obs[0] * 5.0
    cart_y = obs[1] * 4.0
    c = obs[2]
    s = obs[3]

    rel_x = obs[6] * 3.0
    rel_y = obs[7] * 3.0
    d_cart_crate = (rel_x * rel_x + rel_y * rel_y) ** 0.5

    nrel_x = next_obs[6] * 