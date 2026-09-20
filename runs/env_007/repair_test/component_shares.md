| candidate | mean eval reward | episode terminations | success_event active | mean generated reward (last snapshot) | dominant components (share) |
|---|---:|---|---:|---:|---|
| r00_copy | -1.23 | terminated=0, truncated=20 | 0.0000 | -1.411e-06 | dock_speed_penalty 94%, crate_to_dock_progress 6% |
| r01_guidance | -0.40 | terminated=0, truncated=20 | 0.0000 | 7.764e-04 | REPAIR_cart_approach 100% |
| r02_guidance_gentle | -0.10 | terminated=0, truncated=20 | 0.0000 | 1.294e-03 | REPAIR_cart_approach 100% |
| r03_guidance_x3 | 0.34 | terminated=0, truncated=20 | 0.0000 | 7.937e-03 | REPAIR_cart_approach 88%, dock_speed_penalty 6%, REPAIR_crate_progress 4%, soft_contact 1% |
