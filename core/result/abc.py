"""
Vẽ đường cong fitness của các lần chạy GA Scheduling.
Đọc tất cả file fitness_history_*.json trong folder result/
và vẽ từng đường theo field "name" trong mỗi file.

Cách dùng:
    python plot_fitness.py                  # tìm trong ./result/
    python plot_fitness.py --folder path/to/result
    python plot_fitness.py --save           # lưu ảnh thay vì hiện cửa sổ
"""

import json
import glob
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── màu sắc đẹp cho nhiều đường ──────────────────────────────────────────────
COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#000000",
]


def load_histories(folder: str):
    pattern = os.path.join(folder, "fitness_history_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy file fitness_history_*.json trong '{folder}'")

    records = []
    for fpath in files:
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
        name = data.get("name") or os.path.basename(fpath)
        history = data["history"]
        best = data.get("best_fitness", history[-1])
        records.append({"name": name, "history": history, "best": best})

    # Sắp xếp theo best_fitness (tốt nhất — gần 0 — lên trên legend)
    records.sort(key=lambda r: r["best"], reverse=True)
    return records


def plot(records, save_path=None):
    fig, ax = plt.subplots(figsize=(13, 7))

    for i, rec in enumerate(records):
        color = COLORS[i % len(COLORS)]
        gens = list(range(1, len(rec["history"]) + 1))
        ax.plot(
            gens,
            rec["history"],
            color=color,
            linewidth=1.8,
            label=f"{rec['name']}  (best={rec['best']})",
        )
        # đánh dấu điểm cuối
        ax.scatter(gens[-1], rec["history"][-1], color=color, s=40, zorder=5)

    # ── trục & lưới ──────────────────────────────────────────────────────────
    ax.set_xlabel("Thế hệ (Generation)", fontsize=12)
    ax.set_ylabel("Fitness", fontsize=12)
    ax.set_title("Đường cong hội tụ Fitness — GA Scheduling", fontsize=14, fontweight="bold")

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.grid(True, linestyle="--", alpha=0.35)

    # giới hạn trục y: từ min history đến 0 + chút padding
    all_vals = [v for r in records for v in r["history"]]
    ax.set_ylim(min(all_vals) * 1.05, 20)
    ax.set_xlim(1, max(len(r["history"]) for r in records))

    ax.legend(
        loc="lower right",
        fontsize=9,
        framealpha=0.85,
        title="Cấu hình  (best fitness)",
        title_fontsize=9,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Đã lưu đồ thị: {save_path}")
    else:
        plt.show()


def main():
    # Mặc định tìm file trong cùng thư mục với script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Vẽ đường cong fitness GA")
    parser.add_argument(
        "--folder", default=script_dir,
        help="Thư mục chứa các file fitness_history_*.json (mặc định: cùng thư mục script)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Lưu ảnh PNG thay vì hiện cửa sổ"
    )
    parser.add_argument(
        "--out", default="fitness_curves.png",
        help="Tên file ảnh đầu ra khi dùng --save (mặc định: fitness_curves.png)"
    )
    args = parser.parse_args()

    records = load_histories(args.folder)
    print(f"Tìm thấy {len(records)} file:")
    for r in records:
        print(f"  {r['name']:20s}  gen={len(r['history'])}  best={r['best']}")

    save_path = args.out if args.save else None
    plot(records, save_path=save_path)


if __name__ == "__main__":
    main()