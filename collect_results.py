from pathlib import Path

BASE = Path("runs/env_001/ablation_score_only_v4")
OUT = "memory_summary_verified.csv"

rows = []
for seed_dir in sorted(BASE.glob("seed_*")):
    seed = seed_dir.name.split("_")[-1]
    md_file = seed_dir / "memory" / "reward_memory.md"
    if not md_file.exists():
        print(f"Warning: {md_file} not found, skipping seed {seed}")
        continue
    print(f"Processing: {md_file}")

    best_score = -float("inf")
    best_iter = None
    with open(md_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    header_found = False
    for line in lines:
        if line.startswith("|---"):
            header_found = True
            continue
        if not header_found or not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 5:
            continue
        try:
            iter_num = int(parts[1])
            best_val = float(parts[4])   # best 列
        except (ValueError, IndexError):
            continue
        if best_val > best_score:
            best_score = best_val
            best_iter = iter_num

    if best_iter is not None:
        rows.append((int(seed), best_iter, best_score))
        print(f"  -> Seed {seed}: best = {best_score:.2f} at iter {best_iter}")
    else:
        print(f"  -> Seed {seed}: could not parse any data")

with open(OUT, "w", newline="", encoding="utf-8") as f:
    f.write("Seed,Best_Iter,Best_Score\n")
    for seed_int, iter_num, score in sorted(rows, key=lambda x: x[0]):
        f.write(f"{seed_int},{iter_num},{score:.2f}\n")

print(f"\nDone. Summary saved to {OUT}")