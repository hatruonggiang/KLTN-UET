from constraints import count_hard_violations, count_soft_penalties

# ============================================================
# TRỌNG SỐ (WEIGHTS)
# ============================================================

HARD_WEIGHTS = {
    "each_pair_twice":            10000,
    "one_match_per_round":        10000,
    "no_same_stadium_timeslot":    5000,
}

def get_soft_weights(num_teams: int):
    base = {
        # F1 — Công bằng đội
        "no_more_than_3_consecutive": 1,
        "home_away_balance":        5,  # Nghiệp vụ cao + scale nhỏ (2-4 vi phạm)
        "min_rest_days":            1,  # Ảnh hưởng thể lực cầu thủ + khó fix
        "home_distribution":        1,  # Quan trọng nhưng scale trung bình
        "season_edge_balance":      2,  # Ít vi phạm hơn, dễ fix hơn

        # F2 — Chất lượng trận
        "big_match_half_balance":   4,  # Nghiệp vụ cao + khó fix
        "derby_distribution":       3,  # Quan trọng với CĐV + scale nhỏ
        "no_derby_same_round":      3,  # Scale nhỏ (0-2), dễ check
        "no_consec_big_team":       1,  # Có repair riêng → bớt gánh nặng weight
        "big_match_monthly":        2,  # Scale trung bình
        "big_match_alternation":    2,  # Scale trung bình

        # F3 — Vận hành / Phát sóng
        "prime_timeslot_fairness":  3,  # Đài truyền hình → nghiệp vụ cao
        "no_timeslot_overload":     1,  # Scale lớn (có thể 20-50) + dễ fix
        "travel_distance_balance":  10,  # Scale lớn + conflict với min_rest_days
    }

    # if num_teams <= 12:
    #     base.update({
    #         # Giải nhỏ: ít derby + ít big match → tăng các constraint còn lại
    #         "home_away_balance":       6,  # Càng ít đội càng dễ mất cân bằng
    #         "min_rest_days":           5,
    #         "no_derby_same_round":     4,  # Derby hiếm → phải giữ chặt
    #         "home_distribution":       3,
    #         "travel_distance_balance": 1,  # Địa lý nhỏ → ít ý nghĩa
    #     })
    # elif num_teams >= 18:
    #     base.update({
    #         # Giải lớn: nhiều vòng → home_distribution khó hơn
    #         "home_away_balance":       5,
    #         "home_distribution":       4,  # 34 vòng → phân bổ khó hơn nhiều
    #         "big_match_half_balance":  4,  # Nhiều big match → cần kiểm soát chặt
    #         "season_edge_balance":     3,  # Mùa dài → edge quan trọng hơn
    #         "travel_distance_balance": 2,  # Nhiều đội → travel thực sự tốn kém
    #     })

    return base


# ============================================================
# HÀM TÍNH FITNESS
# ============================================================

def calculate_fitness(individual, tournament):
    """Fitness càng cao càng tốt (âm của tổng penalty)."""
    if not individual.matches:
        individual.fitness = -999999
        return individual.fitness

    hard_violations = count_hard_violations(individual, tournament)
    soft_penalties  = count_soft_penalties(individual, tournament)

    soft_weights = get_soft_weights(tournament.num_teams)
    hard_score   = sum(HARD_WEIGHTS.get(k, 1000) * v for k, v in hard_violations.items())
    soft_score   = sum(soft_weights.get(k, 2)    * v for k, v in soft_penalties.items())

    individual.fitness = -(hard_score + soft_score)
    return individual.fitness


def is_valid_schedule(individual, tournament=None):
    """Kiểm tra lịch có hợp lệ không (không vi phạm ràng buộc cứng)."""
    if individual.fitness is None:
        if tournament:
            calculate_fitness(individual, tournament)
        else:
            return False
    return all(v == 0 for v in count_hard_violations(individual, tournament).values())


def get_fitness_details(individual, tournament):
    """Trả về báo cáo chi tiết fitness."""
    hard_violations = count_hard_violations(individual, tournament)
    soft_penalties  = count_soft_penalties(individual, tournament)
    soft_weights    = get_soft_weights(tournament.num_teams)

    hard_details = {
        k: {"violations": v,
            "weight":     HARD_WEIGHTS.get(k, 1000),
            "penalty":    HARD_WEIGHTS.get(k, 1000) * v}
        for k, v in hard_violations.items()
    }
    soft_details = {
        k: {"violations": v,
            "weight":     soft_weights.get(k, 2),
            "penalty":    soft_weights.get(k, 2) * v}
        for k, v in soft_penalties.items()
    }

    hard_total = sum(d["penalty"] for d in hard_details.values())
    soft_total = sum(d["penalty"] for d in soft_details.values())

    return {
        "fitness":       -(hard_total + soft_total),
        "hard_total":    hard_total,
        "soft_total":    soft_total,
        "total_penalty": hard_total + soft_total,
        "hard_details":  hard_details,
        "soft_details":  soft_details,
        "is_valid":      hard_total == 0,
    }


# ============================================================
# TEST / DEBUG
# ============================================================

if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from models import Tournament, Individual, Match

    base_dir   = os.path.dirname(os.path.abspath(__file__))
    tournament = Tournament(os.path.join(base_dir, "../data/teams.json"))

    print("=" * 60)
    print(f"TEST FITNESS FUNCTION")
    print(f"Giải đấu  : {tournament.name}")
    print(f"Số đội    : {tournament.num_teams}")
    print(f"Số vòng   : {tournament.num_rounds}")
    print(f"Timeslots : {tournament.num_timeslots} slot "
          f"(id 0–{tournament.num_timeslots - 1})")
    print(f"Mùa giải  : {tournament.start_date.date()} → {tournament.end_date.date()}")
    print("=" * 60)

    # Kiểm tra nhanh get_match_datetime
    print("\nKiểm tra datetime thật:")
    for slot_id in [0, 2, 4, 6, 12]:
        dt = tournament.get_match_datetime(round_num=1, timeslot_id=slot_id)
        print(f"  Vòng 1 slot {slot_id:>2}: {dt.strftime('%A %d/%m/%Y %Hh%M')}")

    # Dummy matches — dùng timeslot_id trong range mới (0–12)
    dummy_matches = [
        Match(home_team_id=0, away_team_id=1,  round=1, timeslot_id=4),
        Match(home_team_id=2, away_team_id=3,  round=1, timeslot_id=6),
        Match(home_team_id=1, away_team_id=0,  round=2, timeslot_id=2),
    ]

    individual = Individual(dummy_matches)
    details    = get_fitness_details(individual, tournament)

    print(f"\nFitness tổng    : {details['fitness']:,}")
    print(f"Hard penalty    : {details['hard_total']:,}")
    print(f"Soft penalty    : {details['soft_total']:,}")
    print(f"Schedule hợp lệ : {details['is_valid']}\n")

    print("CHI TIẾT RÀNG BUỘC CỨNG:")
    for k, d in details["hard_details"].items():
        print(f"  {k:30s}: vi phạm={d['violations']:2d} | penalty={d['penalty']:>8,d}")

    print("\nCHI TIẾT RÀNG BUỘC MỀM:")
    for k, d in details["soft_details"].items():
        print(f"  {k:30s}: vi phạm={d['violations']:2d} | penalty={d['penalty']:>8,d}")

    print("\n" + "=" * 60)
    print("✅ fitness.py sẵn sàng với timeslot cấu trúc mới (day_offset + hour)")