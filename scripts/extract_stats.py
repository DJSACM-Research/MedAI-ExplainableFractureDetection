"""Extract per-class statistics from JSON artifacts for the paper."""
import json
from collections import Counter, defaultdict

# Conformal results
with open("outputs/conformal_results.json") as f:
    cr = json.load(f)

for alpha_key in ["0.05", "0.1"]:
    r = cr[alpha_key]
    per_class_cov = defaultdict(lambda: {"total": 0, "covered": 0, "correct": 0, "set_sizes": []})
    for s in r["per_sample"]:
        cls = s["true_class"]
        per_class_cov[cls]["total"] += 1
        per_class_cov[cls]["covered"] += int(s["true_covered"])
        per_class_cov[cls]["correct"] += int(s["argmax_correct"])
        per_class_cov[cls]["set_sizes"].append(s["set_size"])
    print(f"\n=== alpha={alpha_key} per-class breakdown ===")
    for cls in sorted(per_class_cov.keys()):
        d = per_class_cov[cls]
        cov = d["covered"] / d["total"] * 100
        acc = d["correct"] / d["total"] * 100
        avg_sz = sum(d["set_sizes"]) / len(d["set_sizes"])
        print(f"  {cls:25s}: n={d['total']:3d}, cov={cov:5.1f}%, acc={acc:5.1f}%, avg_set={avg_sz:.2f}")

# Critic per-class
with open("outputs/critic_evaluation.json") as f:
    ce = json.load(f)

print("\n=== Critic per-class (all confirmed) ===")
per_class_critic = defaultdict(lambda: {"total": 0, "correct": 0, "verdicts": Counter()})
for s in ce["per_sample"]:
    cls = s["true_class"]
    per_class_critic[cls]["total"] += 1
    per_class_critic[cls]["correct"] += int(s["is_correct"])
    per_class_critic[cls]["verdicts"][s["critic_verdict"]] += 1

for cls in sorted(per_class_critic.keys()):
    d = per_class_critic[cls]
    acc = d["correct"] / d["total"] * 100
    print(f"  {cls:25s}: n={d['total']:3d}, acc={acc:5.1f}%, verdicts={dict(d['verdicts'])}")

# Safety cases detail at alpha=0.1
print("\n=== Safety cases (alpha=0.05) ===")
for s in cr["0.05"]["safety_cases"]:
    print(f"  true={s['true_class']}, pred={s['argmax_pred']}, conformal_set={s['conformal_set']}")

print("\n=== Safety cases (alpha=0.1) ===")
for s in cr["0.1"]["safety_cases"]:
    print(f"  true={s['true_class']}, pred={s['argmax_pred']}, conformal_set={s['conformal_set']}")

# Misclassified cases
print("\n=== All misclassified cases ===")
for s in cr["0.1"]["per_sample"]:
    if not s["argmax_correct"]:
        print(f"  true={s['true_class']:25s}, pred={s['argmax_pred']:25s}, covered={s['true_covered']}, set={s['conformal_set']}")

# Confusion matrix
print("\n=== Confusion pairs (true -> pred) ===")
confusion = Counter()
for s in cr["0.1"]["per_sample"]:
    confusion[(s["true_class"], s["argmax_pred"])] += 1
for (t, p), c in sorted(confusion.items()):
    if t != p:
        print(f"  {t:25s} -> {p:25s}: {c}")
