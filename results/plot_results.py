# plot_results.py — đặt cùng thư mục results/
import json, glob, os
import matplotlib.pyplot as plt

result_dir = os.path.dirname(os.path.abspath(__file__))

# ── Hình 4.3: Pareto front F1 vs F2 ───────────────────────────
# Dùng [0-9] để chỉ lấy file bắt đầu bằng số đội (ví dụ pareto_14teams) 
pareto_files = sorted(glob.glob(os.path.join(result_dir, "pareto_[0-9]*teams_*.json")))
if pareto_files:
    with open(pareto_files[-1]) as f:   # lấy file mới nhất
        pts = json.load(f)
    f1 = [p["f1"] for p in pts]
    f2 = [p["f2"] for p in pts]
    f3 = [p["f3"] for p in pts]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].scatter(f1, f2, c=f3, cmap="viridis", s=60, edgecolors="k", linewidths=0.5)
    axes[0].set_xlabel("F1 (Công bằng đội)")
    axes[0].set_ylabel("F2 (Chất lượng trận)")
    axes[0].set_title("Pareto front: F1 vs F2")
    sm = plt.cm.ScalarMappable(cmap="viridis")
    sm.set_array(f3)
    plt.colorbar(sm, ax=axes[0], label="F3")

    axes[1].scatter(f1, f3, c=f2, cmap="plasma", s=60, edgecolors="k", linewidths=0.5)
    axes[1].set_xlabel("F1 (Công bằng đội)")
    axes[1].set_ylabel("F3 (Vận hành/Phát sóng)")
    axes[1].set_title("Pareto front: F1 vs F3")
    sm2 = plt.cm.ScalarMappable(cmap="plasma")
    sm2.set_array(f2)
    plt.colorbar(sm2, ax=axes[1], label="F2")

    axes[2].scatter(f2, f3, c=f1, cmap="coolwarm", s=60, edgecolors="k", linewidths=0.5)
    axes[2].set_xlabel("F2 (Chất lượng trận)")
    axes[2].set_ylabel("F3 (Vận hành/Phát sóng)")
    axes[2].set_title("Pareto front: F2 vs F3")
    sm3 = plt.cm.ScalarMappable(cmap="coolwarm")
    sm3.set_array(f1)
    plt.colorbar(sm3, ax=axes[2], label="F1")

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "fig_pareto_front.png"), dpi=150)
    print("✅ fig_pareto_front.png")

# ── Hình 4.4: So sánh kích thước tập Pareto theo số đội ────────
plt.figure(figsize=(10, 6))
team_counts = [8, 10, 12, 14, 16, 18, 20]
found_any = False

for n in team_counts:
    # Tìm file log pareto_size mới nhất cho từng số lượng đội
    pattern = os.path.join(result_dir, f"pareto_size_{n}teams_*.json")
    files = sorted(glob.glob(pattern))
    if files:
        with open(files[-1]) as f:
            data = json.load(f)
        gens = [d["gen"] for d in data]
        # Sử dụng archive_size vì đây là số lượng nghiệm tối ưu thực sự được giữ lại
        sizes = [d["archive_size"] for d in data]
        plt.plot(gens, sizes, linewidth=2, label=f"{n} Đội")
        found_any = True

if found_any:
    plt.xlabel("Thế hệ")
    plt.ylabel("Kích thước tập Pareto (Số lượng nghiệm)")
    plt.title("Hình 4.4: Sự thay đổi kích thước tập Pareto theo số thế hệ")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "fig_convergence.png"), dpi=150)
    print("✅ fig_convergence.png")
    print("✅ fig_convergence.png (Biểu đồ so sánh đa quy mô)")

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "fig_convergence.png"), dpi=150)
    print("✅ fig_convergence.png")

plt.show()