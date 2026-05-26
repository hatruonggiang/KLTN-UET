from collections import defaultdict
import math


# ============================================================
# RÀNG BUỘC CỨNG (HARD CONSTRAINTS)
# ============================================================

def check_each_pair_meets_twice(matches, num_teams):
    """Mỗi cặp đội gặp nhau đúng 2 lần (1 nhà - 1 khách)."""
    violations = 0
    pair_count = defaultdict(int)

    for m in matches:
        pair_count[(m.home_team_id, m.away_team_id)] += 1

    for i in range(num_teams):
        for j in range(num_teams):
            if i != j:
                count = pair_count.get((i, j), 0)
                if count != 1:
                    violations += abs(count - 1)

    return violations


def check_one_match_per_round(matches, num_teams, num_rounds):
    violations = 0
    team_round = defaultdict(lambda: defaultdict(int))
    for m in matches:
        team_round[m.home_team_id][m.round] += 1
        team_round[m.away_team_id][m.round] += 1

    for team_id in range(num_teams):
        for r in range(1, num_rounds + 1):
            count = team_round[team_id][r]
            if count > 1:
                violations += count - 1
    return violations


def check_no_same_stadium_same_timeslot(matches):
    """Không hai trận cùng sân + cùng timeslot trong cùng vòng."""
    violations = 0
    seen = defaultdict(int)
    for m in matches:
        key = (m.home_team_id, m.round, m.timeslot_id)
        seen[key] += 1

    for count in seen.values():
        if count > 1:
            violations += count - 1
    return violations


def check_no_more_than_3_consecutive(matches, num_teams, num_rounds):
    """Không hơn 3 trận sân nhà hoặc sân khách liên tiếp."""
    violations = 0
    schedule = defaultdict(dict)

    for m in matches:
        schedule[m.home_team_id][m.round] = 'H'
        schedule[m.away_team_id][m.round] = 'A'

    for team_id in range(num_teams):
        consecutive = 1
        prev = None
        for r in range(1, num_rounds + 1):
            current = schedule[team_id].get(r)
            if current is None:
                consecutive = 1
                prev = None
                continue
            if current == prev:
                consecutive += 1
                if consecutive > 3:
                    violations += 1
            else:
                consecutive = 1
            prev = current
    return violations


def count_hard_violations(individual, tournament):
    matches  = individual.matches
    n_teams  = tournament.num_teams
    n_rounds = tournament.num_rounds

    return {
        "each_pair_twice":           check_each_pair_meets_twice(matches, n_teams),
        "one_match_per_round":       check_one_match_per_round(matches, n_teams, n_rounds),
        "no_same_stadium_timeslot":  check_no_same_stadium_same_timeslot(matches),
        # "no_more_than_3_consecutive": check_no_more_than_3_consecutive(matches, n_teams, n_rounds),
    }


# ============================================================
# RÀNG BUỘC MỀM (SOFT CONSTRAINTS)
# ============================================================

def soft_travel_distance_balance(matches, num_teams, tournament):
    """
    Tính tổng quãng đường di chuyển THỰC TẾ của mỗi đội trong mùa giải.
    
    Logic: đội luôn xuất phát từ sân nhà. Khi đá away thì di chuyển đến
    sân đối phương. Chuỗi away liên tiếp thì đi thẳng từ sân này sang sân kia,
    không về nhà giữa chừng. Sau chuỗi away thì về nhà.
    
    Ví dụ đội A (nhà tại city_A):
      Vòng 1: home → không di chuyển
      Vòng 2: away tại B → A→B
      Vòng 3: away tại C → B→C  (không về nhà)
      Vòng 4: home → C→A        (về nhà)
      Vòng 5: away tại D → A→D
      Vòng 6: home → D→A
    Tổng: A→B + B→C + C→A + A→D + D→A
    """
    total_dist = defaultdict(float)

    # Sắp xếp tất cả trận theo vòng
    sorted_matches = sorted(matches, key=lambda m: m.round)

    for team_id in range(num_teams):
        home_team    = tournament.get_team(team_id)
        current_city = home_team  # bắt đầu tại sân nhà

        # Lấy các trận của đội này, theo thứ tự vòng
        team_matches = [
            m for m in sorted_matches
            if m.home_team_id == team_id or m.away_team_id == team_id
        ]

        for m in team_matches:
            if m.away_team_id == team_id:
                # Đá away: di chuyển từ vị trí hiện tại đến sân đối phương
                destination  = tournament.get_team(m.home_team_id)
                total_dist[team_id] += current_city.distance_to(destination)
                current_city = destination
            else:
                # Đá home: nếu đang ở xa thì về nhà trước
                if current_city != home_team:
                    total_dist[team_id] += current_city.distance_to(home_team)
                    current_city = home_team
                # Đang ở nhà rồi thì không tốn chi phí

    # Về nhà sau trận cuối nếu đang ở xa
    for team_id in range(num_teams):
        # (đã xử lý trong loop trên khi gặp trận home tiếp theo,
        #  nhưng cần xử lý trường hợp mùa kết thúc bằng away)
        pass

    # Tính penalty: phạt đội có tổng đường đi lệch nhiều so với trung bình
    if not total_dist:
        return 0

    dists = [total_dist.get(i, 0.0) for i in range(num_teams)]
    mean  = sum(dists) / max(num_teams, 1)

    if mean < 1:
        return 0   # distance_to luôn = 0 → không có dữ liệu địa lý

    std      = math.sqrt(sum((d - mean) ** 2 for d in dists) / max(num_teams, 1))
    outliers = sum(1 for d in dists if abs(d - mean) > mean * 0.20)

    return int(std / mean * 100) + outliers


def soft_home_away_balance(matches, num_teams):
    penalty    = 0
    home_count = defaultdict(int)
    for m in matches:
        home_count[m.home_team_id] += 1
    target_home = num_teams - 1
    for team_id in range(num_teams):
        penalty += abs(home_count[team_id] - target_home)
    return penalty


def soft_home_distribution(matches, num_teams, num_rounds, num_segments=4):
    """Phân bổ sân nhà đều theo các đoạn của mùa giải."""
    penalty          = 0
    segment_size     = max(1, num_rounds // num_segments)
    ideal_per_segment = (num_teams - 1) / num_segments

    home_per_segment = defaultdict(lambda: defaultdict(int))
    for m in matches:
        segment = (m.round - 1) // segment_size
        home_per_segment[m.home_team_id][segment] += 1

    for team_id in range(num_teams):
        for seg in range(num_segments):
            deviation = abs(home_per_segment[team_id][seg] - ideal_per_segment)
            if deviation > 1.5:
                penalty += deviation ** 2
    return int(penalty)


def soft_min_rest_days(matches, num_teams, tournament, min_rest_hours=24):
    """
    Tối thiểu min_rest_hours giờ nghỉ giữa 2 trận liên tiếp của mỗi đội.
    Dùng datetime thật từ tournament.get_match_datetime().
    Mặc định 24h = 1 ngày.
    """
    penalty       = 0
    team_schedule = defaultdict(list)

    for m in matches:
        dt = tournament.get_match_datetime(m.round, m.timeslot_id)
        team_schedule[m.home_team_id].append(dt)
        team_schedule[m.away_team_id].append(dt)

    for team_id in range(num_teams):
        sorted_dates = sorted(team_schedule[team_id])
        for i in range(1, len(sorted_dates)):
            diff_hours = (sorted_dates[i] - sorted_dates[i - 1]).total_seconds() / 3600
            if diff_hours < min_rest_hours:
                penalty += 1

    return penalty


def soft_derby_distribution(matches, tournament):
    """Derby không nên rơi cùng nửa mùa."""
    penalty   = 0
    mid       = tournament.num_rounds // 2
    
    # Nhóm các trận đấu theo cặp đội (không phân biệt sân nhà/khách) O(M)
    pair_map = defaultdict(list)
    for m in matches:
        pair = tuple(sorted([m.home_team_id, m.away_team_id]))
        pair_map[pair].append(m)

    for pair, ms in pair_map.items():
        if len(ms) < 2: continue
        
        # Chỉ kiểm tra nếu là Derby
        t1 = tournament.get_team(pair[0])
        t2 = tournament.get_team(pair[1])
        if t1.city == t2.city:
            m1, m2 = ms[0], ms[1]
            # Nếu cả lượt đi và lượt về đều nằm cùng một nửa mùa giải
            if (m1.round <= mid) == (m2.round <= mid):
                penalty += 2
    return penalty


def soft_no_derby_same_round(matches, tournament, max_derby_per_round=1):
    """Tối đa max_derby_per_round derby trong cùng một vòng."""
    penalty         = 0
    derby_per_round = defaultdict(int)
    for m in matches:
        if tournament.get_team(m.home_team_id).city == tournament.get_team(m.away_team_id).city:
            derby_per_round[m.round] += 1

    for count in derby_per_round.values():
        if count > max_derby_per_round:
            penalty += count - max_derby_per_round
    return penalty


def soft_season_edge_balance(matches, num_teams, num_rounds, edge_size=5, max_consec_away=2):
    """Tránh chuỗi sân khách ở đầu và cuối mùa."""
    penalty    = 0
    edge_start = set(range(1, edge_size + 1))
    edge_end   = set(range(num_rounds - edge_size + 2, num_rounds + 1))
    schedule   = defaultdict(dict)

    for m in matches:
        schedule[m.home_team_id][m.round] = 'H'
        schedule[m.away_team_id][m.round] = 'A'

    for team_id in range(num_teams):
        for edge in (edge_start, edge_end):
            away_streak = 0
            for r in sorted(edge):
                if schedule[team_id].get(r) == 'A':
                    away_streak += 1
                    if away_streak >= max_consec_away:
                        penalty += 1
                        break
                else:
                    away_streak = 0
    return penalty


def soft_no_consec_big_team(matches, num_teams, tournament, threshold=8):
    """
    Ràng buộc mềm: Hạn chế một đội phải gặp 2 đối thủ mạnh liên tiếp.
    
    Args:
        matches: Danh sách các trận đấu hiện tại.
        num_teams: Tổng số đội trong giải.
        tournament: Đối tượng Tournament chứa dữ liệu champ_potential.
        threshold: Ngưỡng điểm tiềm năng để xác định "đội bóng lớn" (mặc định là 8).
        
    Returns:
        int: Số lần vi phạm (số cặp vòng đấu liên tiếp gặp đối thủ mạnh).
    """
    penalty  = 0
    schedule = defaultdict(dict) # Map: team_id -> {round -> (opponent_id, is_home)}
    
    # Bước 1: Chuyển đổi danh sách trận đấu thành cấu trúc lịch trình theo từng đội
    for m in matches:
        schedule[m.home_team_id][m.round] = (m.away_team_id, True)
        schedule[m.away_team_id][m.round] = (m.home_team_id, False)

    # Bước 2: Duyệt qua từng đội để kiểm tra các vòng đấu liên tiếp
    for team_id in range(num_teams):
        rounds = sorted(schedule[team_id].keys())
        for i in range(len(rounds) - 1):
            r1, r2 = rounds[i], rounds[i + 1] # Lấy hai vòng đấu xuất hiện trong lịch
            
            # Chỉ xử lý nếu hai vòng đấu này thực sự sát nhau (ví dụ: vòng 5 và vòng 6)
            if r2 != r1 + 1:
                continue
                
            # Lấy thông tin đối thủ tại hai thời điểm đó
            opp1, _ = schedule[team_id][r1]
            opp2, _ = schedule[team_id][r2]
            
            # Bước 3: Kiểm tra nếu cả hai đối thủ đều là "ông lớn" dựa trên ngưỡng threshold
            if (tournament.get_champ_potential(opp1) >= threshold and
                    tournament.get_champ_potential(opp2) >= threshold):
                penalty += 1
    return penalty


def soft_big_match_half_balance(matches, num_rounds, tournament, threshold=8):
    penalty   = 0
    mid       = num_rounds // 2
    big_count = defaultdict(lambda: [0, 0])

    for m in matches:
        if (tournament.get_champ_potential(m.home_team_id) >= threshold and
                tournament.get_champ_potential(m.away_team_id) >= threshold):
            half = 0 if m.round <= mid else 1
            big_count[m.home_team_id][half] += 1
            big_count[m.away_team_id][half] += 1

    for counts in big_count.values():
        diff = abs(counts[0] - counts[1])
        if diff > 1:
            penalty += diff - 1
    return penalty


def soft_big_match_monthly(matches, tournament, num_rounds, threshold=8, rounds_per_month=4):
    penalty       = 0
    big_per_month = defaultdict(int)
    for m in matches:
        if (tournament.get_champ_potential(m.home_team_id) >= threshold and
                tournament.get_champ_potential(m.away_team_id) >= threshold):
            month = (m.round - 1) // rounds_per_month
            big_per_month[month] += 1

    if big_per_month:
        counts = list(big_per_month.values())
        avg    = sum(counts) / len(counts)
        penalty = int(sum((c - avg) ** 2 for c in counts))
    return penalty


def soft_big_match_alternation(matches, num_rounds, tournament, threshold=8, max_no_big_streak=1):
    penalty = 0
    has_big = set()
    for m in matches:
        if (tournament.get_champ_potential(m.home_team_id) >= threshold and
                tournament.get_champ_potential(m.away_team_id) >= threshold):
            has_big.add(m.round)

    streak = 0
    for r in range(1, num_rounds + 1):
        if r in has_big:
            streak = 0
        else:
            streak += 1
            if streak > max_no_big_streak:
                penalty += 1
    return penalty


def soft_prime_timeslot_fairness(matches, num_teams, num_rounds, tournament,
                                  prime_hour=19, min_prime_per_half=2):
    penalty     = 0
    mid         = num_rounds // 2
    prime_count = defaultdict(lambda: [0, 0])

    for m in matches:
        ts = tournament.get_timeslot(m.timeslot_id)
        if ts["hour"] >= prime_hour:
            half = 0 if m.round <= mid else 1
            prime_count[m.home_team_id][half] += 1
            prime_count[m.away_team_id][half] += 1

    # Tính tổng prime slots có thể có
    total_prime_slots = sum(
        1 for m in matches
        if tournament.get_timeslot(m.timeslot_id)["hour"] >= prime_hour
    )
    
    if total_prime_slots == 0:
        # Không có prime slot nào trong tournament → constraint vô nghĩa
        # Chuyển sang đo độ công bằng timeslot tổng quát
        slot_count = defaultdict(lambda: defaultdict(int))
        for m in matches:
            slot_count[m.home_team_id][m.timeslot_id] += 1
            slot_count[m.away_team_id][m.timeslot_id] += 1
        # Phạt đội bị dồn vào ít loại timeslot
        for team_id in range(num_teams):
            unique_slots = len(slot_count[team_id])
            if unique_slots < tournament.num_timeslots // 2:
                penalty += tournament.num_timeslots // 2 - unique_slots
        return penalty

    # Logic gốc — nhưng scale min_prime_per_half theo num_rounds
    min_prime = max(1, num_rounds // 8)   # thay vì hardcode = 2
    for team_id in range(num_teams):
        for half in range(2):
            deficit = max(0, min_prime - prime_count[team_id][half])
            penalty += deficit * 2    # nhân 2 để tăng signal
    return penalty


def soft_no_timeslot_overload(matches, max_matches_per_slot=3):
    """Không quá max_matches_per_slot trận trong cùng round + timeslot."""
    penalty = 0
    count   = defaultdict(int)
    for m in matches:
        count[(m.round, m.timeslot_id)] += 1

    for c in count.values():
        if c > max_matches_per_slot:
            penalty += c - max_matches_per_slot
    return penalty


# ============================================================
# TỔNG HỢP
# ============================================================

def count_soft_penalties(individual, tournament):
    matches  = individual.matches
    n_teams  = tournament.num_teams
    n_rounds = tournament.num_rounds

    return {
        "no_more_than_3_consecutive": check_no_more_than_3_consecutive(matches, n_teams, n_rounds),
        "home_away_balance":       soft_home_away_balance(matches, n_teams),
        "home_distribution":       soft_home_distribution(matches, n_teams, n_rounds),
        "min_rest_days":           soft_min_rest_days(matches, n_teams, tournament, min_rest_hours=72),
        "derby_distribution":      soft_derby_distribution(matches, tournament),
        "no_derby_same_round":     soft_no_derby_same_round(matches, tournament, max_derby_per_round=1),
        "season_edge_balance":     soft_season_edge_balance(matches, n_teams, n_rounds, edge_size=5, max_consec_away=2),
        "no_consec_big_team":      soft_no_consec_big_team(matches, n_teams, tournament, threshold=8),
        "big_match_half_balance":  soft_big_match_half_balance(matches, n_rounds, tournament, threshold=8),
        "big_match_monthly":       soft_big_match_monthly(matches, tournament, n_rounds, threshold=8, rounds_per_month=4),
        "big_match_alternation":   soft_big_match_alternation(matches, n_rounds, tournament, threshold=8, max_no_big_streak=1),
        "prime_timeslot_fairness": soft_prime_timeslot_fairness(matches, n_teams, n_rounds, tournament, prime_hour=19, min_prime_per_half=2),
        "no_timeslot_overload":    soft_no_timeslot_overload(matches, max_matches_per_slot=3),
        "travel_distance_balance": soft_travel_distance_balance(matches, n_teams, tournament),
    }