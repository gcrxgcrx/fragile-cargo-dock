| candidate | mean eval reward | episode terminations | success_event active | mean generated reward (last snapshot) | dominant components (share) |
|---|---:|---|---:|---:|---|
| L0_cand_00 | -67.40 | terminated=13, truncated=7 | n/a | -1.502e-02 | out_of_bounds_penalty 59%, action_smoothness 41% |
| L0_cand_02 | 1.53 | terminated=0, truncated=20 | n/a | 6.132e-01 | crate_docking_quality 94%, soft_contact_penalty 3%, action_smoothness 2%, crate_to_dock_progress 1% |
| L0_cand_11 | -2.26 | terminated=0, truncated=20 | n/a | 9.011e-01 | crate_docking_quality 72%, crate_dock_proximity 27%, action_smoothness_penalty 0%, out_of_bounds_penalty 0% |
| L0_cand_13 | -1.86 | terminated=0, truncated=20 | n/a | 4.192e-01 | crate_docking_quality 100%, action_smoothness 0% |
| L2_cand_00 | -1.23 | terminated=0, truncated=20 | 0.0000 | -1.411e-06 | dock_speed_penalty 94%, crate_to_dock_progress 6% |
| L2_cand_04 | -0.93 | terminated=0, truncated=20 | 0.0000 | -8.618e-03 | action_penalty 100% |
| L2_cand_05 | -1.33 | terminated=0, truncated=20 | 0.0000 | 0.000e+00 | (none active) |
| L2_cand_14 | -1.29 | terminated=0, truncated=20 | 0.0000 | 1.979e-03 | crate_dock_progress 100% |
