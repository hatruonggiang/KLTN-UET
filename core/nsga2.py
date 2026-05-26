import random
import time
import copy
import math
from collections import defaultdict, deque
from itertools import combinations_with_replacement

from models import Match, Individual
from constraints import count_hard_violations, count_soft_penalties
from ga import (
    create_population,
    repair_individual,
    repair_no_consec_big_team,
    repair_big_match_alternation,
    crossover,
    mutate,
    local_search_timeslot,
    local_search_round_swap,
    local_search_big_match_chain,
    _create_round_robin_individual,
    _create_biased_individual,
    shuffle_round_mutation,
    swap_team_identities_mutation,
    cyclic_shift_mutation,
)

KEYS_F1 = [
    "home_away_balance",      # Cân bằng số trận nhà/khách
    "home_distribution",      # Phân bổ trận nhà đều theo mùa
    "min_rest_days",          # Đủ ngày nghỉ giữa 2 trận
    "season_edge_balance",    # Tránh chuỗi sân khách đầu/cuối mùa
    "no_more_than_3_consecutive",
]

KEYS_F2 = [
    "derby_distribution",     # Derby không cùng nửa mùa
    "no_derby_same_round",    # Tối đa 1 derby/vòng
    "no_consec_big_team",     # Không 2 big match liên tiếp (w=0 → không penalty)
    "big_match_half_balance", # Big match đều 2 nửa mùa
    "big_match_monthly",      # Big match đều theo tháng
    "big_match_alternation",  # Không bỏ trống quá 1 vòng không có big match
]

KEYS_F3 = [
    "prime_timeslot_fairness",  # Công bằng giờ vàng
    "no_timeslot_overload",     # Không quá nhiều trận/slot
    "travel_distance_balance",  # [L] Chuyển từ F1 → F3 để tạo conflict thật
]

# Ghi chú conflict:
# F1 ↔ F2: home_distribution (F1) ↔ big_match_half_balance (F2)
#           → trải đều home không đảm bảo big match đều 2 nửa
# F1 ↔ F3: min_rest_days (F1) ↔ travel_distance_balance (F3)
#           → nghỉ đủ ngày → ít trận xa → travel không cân bằng
# F2 ↔ F3: big_match_alternation (F2) ↔ prime_timeslot_fairness (F3)
#           → trải đều big match → không phải lúc nào cũng prime slot
# F2 ↔ F3: big_match_monthly (F2) ↔ travel_distance_balance (F3)
#           → trải đều big match theo tháng ↔ travel cân bằng

# [B] Interval tính Hypervolume (thế hệ)
HV_INTERVAL = 5

# [I] Epsilon cho epsilon-dominance trong archive
# Tăng từ 0.05 → 0.10 để archive giữ nhiều nghiệm đa dạng hơn sau khi tách F
ARCHIVE_EPSILON = 0.02


# ============================================================
# TRỌNG SỐ MỀM — lấy trực tiếp từ fitness.py
# ============================================================

from fitness import get_soft_weights as _get_soft_weights_ext


def _get_weights(num_teams: int) -> dict:
    """
    Wrapper lấy soft weights từ fitness.py.
    Đảm bảo nsga2.py và ga.py dùng CÙNG bộ weights.
    
    """
    return _get_soft_weights_ext(num_teams)


# ============================================================
# AUTO-SCALE THAM SỐ THEO SỐ ĐỘI
# ============================================================

def _auto_params(num_teams: int) -> dict:
    return {
        "population_size"        : 250,
        "num_generations"        : 300,
        "crossover_rate"         : 0.90,
        "mutation_rate"          : 0.20,
        "mut_rate_max"           : 0.55,
        "tournament_size"        : 6,
        "tournament_size_max"    : 12,
        "stagnation_patience"    : 92,
        "stagnation_tol"         : 1e-6,
        "hv_window"              : 5,
        "aggressive_after"       : 30,
        "inject_ratio"           : 0.28,
        "front0_bloat_thresh"    : 0.90,
        "explore_ratio"          : 0.15,
        "local_search_interval"  : 4,
        "local_search_top_k"     : 8,
        "restart_limit"          : 40,
        "diversity_threshold"    : 0.40,
        "multi_mutation_prob"    : 0.25,
        "ls_max_iter_timeslot"   : 25,
        "ls_max_iter_round"      : 15,
        "ls_max_iter_chain"      : 20,
        "archive_max_size"       : 250,
        "archive_n_niches"       : 10,
        "archive_epsilon"        : 0.02,
        "oversample_factor"      : 1.5,
        "oversample_after"       : 50,
        "rank_select_ratio"      : 0.25,
        "ref_point_divisions"    : 4,
    }


def build_params(num_teams: int, overrides: dict = None) -> dict:
    params = _auto_params(num_teams)
    if overrides:
        params.update(overrides)
    return params


# ============================================================
# [L] TÍNH OBJECTIVES — phân nhóm mới tạo conflict thật
# ============================================================

def compute_objectives(individual, tournament):
    """
    Gán individual.objectives = (F1, F2, F3) và individual.is_feasible.

    [L] Phân nhóm mới:
      F1 = home_away_balance + home_distribution + min_rest_days + season_edge_balance
      F2 = derby_distribution + no_derby_same_round + no_consec_big_team
           + big_match_half_balance + big_match_monthly + big_match_alternation
      F3 = prime_timeslot_fairness + no_timeslot_overload + travel_distance_balance

    Conflict thật được tạo ra:
      • F1 ↔ F2: home_distribution ↔ big_match_half_balance
      • F1 ↔ F3: min_rest_days ↔ travel_distance_balance
      • F2 ↔ F3: big_match_alternation ↔ prime_timeslot_fairness
                  big_match_monthly ↔ travel_distance_balance

    Hard violations → is_feasible = False, objectives tăng cao để không dominate.
    individual.fitness = -(F1+F2+F3) để tương thích ga.py.
    """
    hard    = count_hard_violations(individual, tournament)
    h_total = sum(hard.values())

    if h_total > 0:
        base = h_total * 1_000
        individual.objectives  = (base + 3000, base + 2000, base + 1000)
        individual.is_feasible = False
        individual.fitness     = -(base + 6000)
        return individual.objectives

    W    = _get_weights(tournament.num_teams)
    soft = count_soft_penalties(individual, tournament)

    # [L] Tổng hợp theo nhóm mới
    f1 = sum(W.get(k, 1) * soft.get(k, 0) for k in KEYS_F1)
    f2 = sum(W.get(k, 1) * soft.get(k, 0) for k in KEYS_F2)
    f3 = sum(W.get(k, 1) * soft.get(k, 0) for k in KEYS_F3)

    individual.objectives  = (f1, f2, f3)
    individual.is_feasible = True
    individual.fitness     = -(f1 + f2 + f3)
    return individual.objectives


# ============================================================
# DOMINANCE & NON-DOMINATED SORT
# ============================================================

def _dominates(obj_a, obj_b):
    """a dominates b: a[i] <= b[i] với mọi i, và a[i] < b[i] ít nhất một i (MIN)."""
    at_least_one = False
    for a, b in zip(obj_a, obj_b):
        if a > b:
            return False
        if a < b:
            at_least_one = True
    return at_least_one


def _epsilon_dominates(obj_a, obj_b, eps=ARCHIVE_EPSILON):
    """
    [I] a ε-dominates b:
    a[i] <= b[i] * (1 + eps) với mọi i, tốt hơn ít nhất 1 chiều bởi eps.
    Buộc nghiệm trong archive phải đủ khác nhau.
    """
    at_least_one = False
    for a, b in zip(obj_a, obj_b):
        if a > b * (1 + eps):
            return False
        if a < b * (1 - eps):
            at_least_one = True
    return at_least_one


def fast_non_dominated_sort(population):
    """
    Phân loại quần thể thành fronts F0, F1, F2, ...
    Feasible luôn dominate infeasible.
    Infeasible xếp riêng theo tổng objectives tăng dần.
    """
    feasible   = [ind for ind in population if getattr(ind, "is_feasible", False)]
    infeasible = [ind for ind in population if not getattr(ind, "is_feasible", False)]

    def _sort_group(group):
        n = len(group)
        if n == 0:
            return []
        dom_count = [0] * n
        dom_set   = [[] for _ in range(n)]
        fronts    = [[]]

        for i in range(n):
            for j in range(i + 1, n):
                oi, oj = group[i].objectives, group[j].objectives
                if _dominates(oi, oj):
                    dom_set[i].append(j)
                    dom_count[j] += 1
                elif _dominates(oj, oi):
                    dom_set[j].append(i)
                    dom_count[i] += 1

        for i in range(n):
            if dom_count[i] == 0:
                group[i].rank = 0
                fronts[0].append(i)

        cur = 0
        while fronts[cur]:
            nxt = []
            for i in fronts[cur]:
                for j in dom_set[i]:
                    dom_count[j] -= 1
                    if dom_count[j] == 0:
                        group[j].rank = cur + 1
                        nxt.append(j)
            cur += 1
            fronts.append(nxt)

        return [[group[i] for i in f] for f in fronts if f]

    feasible_fronts = _sort_group(feasible)

    if infeasible:
        base_rank = len(feasible_fronts)
        infeasible.sort(key=lambda ind: sum(ind.objectives))
        for i, ind in enumerate(infeasible):
            ind.rank = base_rank + i
        infeasible_fronts = [[ind] for ind in infeasible]
    else:
        infeasible_fronts = []

    return feasible_fronts + infeasible_fronts


def crowding_distance_assignment(front):
    """
    [C] Normalized crowding distance — tránh F1 lấn át F2/F3.
    Chia raw distance cho span của từng objective.
    """
    n = len(front)
    if n == 0:
        return
    num_obj = len(front[0].objectives)

    for ind in front:
        ind.crowding_distance = 0.0

    for m in range(num_obj):
        front.sort(key=lambda ind: ind.objectives[m])
        front[0].crowding_distance  = float('inf')
        front[-1].crowding_distance = float('inf')

        global_max = front[-1].objectives[m]
        global_min = front[0].objectives[m]
        norm_factor = (global_max - global_min) or 1e-9

        for i in range(1, n - 1):
            raw_dist = (front[i + 1].objectives[m] - front[i - 1].objectives[m])
            front[i].crowding_distance += raw_dist / norm_factor


# ============================================================
# [K] REFERENCE POINTS (NSGA-III style)
# ============================================================

def generate_reference_points(n_obj=3, n_divisions=4):
    """
    [K] Tạo reference points đều trên simplex chuẩn.
    [L] Sau khi tách objectives, Pareto front rộng hơn nên ref points
        quan trọng hơn để đảm bảo coverage đều.
    """
    points = []
    for combo in combinations_with_replacement(range(n_divisions + 1), n_obj):
        if sum(combo) == n_divisions:
            points.append(tuple(c / n_divisions for c in combo))
    return points


def associate_to_reference_points(population, ref_points):
    """
    [K] Gán mỗi cá thể vào reference point gần nhất (sau normalize).
    Trả về dict {ref_point_idx: count}.
    """
    if not population or not ref_points:
        return defaultdict(int)

    objs = [ind.objectives for ind in population]
    mins = [min(o[i] for o in objs) for i in range(3)]
    maxs = [max(o[i] for o in objs) for i in range(3)]
    spans = [max(mx - mn, 1e-9) for mn, mx in zip(mins, maxs)]

    niche_count = defaultdict(int)
    for ind in population:
        norm = tuple((ind.objectives[i] - mins[i]) / spans[i] for i in range(3))
        closest = min(
            range(len(ref_points)),
            key=lambda j: sum((norm[i] - ref_points[j][i]) ** 2 for i in range(3))
        )
        ind._ref_point = closest
        niche_count[closest] += 1

    return niche_count


# ============================================================
# SELECTION — [A] Adaptive tournament + [H] rank-based + [K] ref-point
# ============================================================

def nsga2_selection(population, k=2, niche_count=None):
    """
    [K] k-way tournament selection với niche awareness.
    Ưu tiên cá thể thuộc niche ít được cover hơn khi rank bằng nhau.
    """
    contestants = random.sample(population, min(k, len(population)))
    best = contestants[0]
    for c in contestants[1:]:
        c_niche = niche_count.get(getattr(c, '_ref_point', -1), 0) if niche_count else 0
        b_niche = niche_count.get(getattr(best, '_ref_point', -1), 0) if niche_count else 0

        if c.rank < best.rank:
            best = c
        elif c.rank == best.rank:
            if c_niche < b_niche:
                best = c
            elif c_niche == b_niche and c.crowding_distance > best.crowding_distance:
                best = c
    return best


def rank_based_selection(population):
    """
    [H] Selection theo rank fitness tuyến tính.
    Cá thể rank Pareto nhỏ có xác suất chọn cao hơn.
    """
    sorted_pop = sorted(
        population,
        key=lambda ind: (ind.rank, -getattr(ind, "crowding_distance", 0.0))
    )
    n = len(sorted_pop)
    weights = list(range(n, 0, -1))
    return random.choices(sorted_pop, weights=weights, k=1)[0]


def _select_parent(population, tourn_k, rank_ratio, niche_count=None):
    """[A][H][K] Chọn parent: dùng rank-based hoặc tournament theo tỷ lệ rank_ratio."""
    if random.random() < rank_ratio:
        return rank_based_selection(population)
    return nsga2_selection(population, tourn_k, niche_count)


def _adaptive_tournament_k(base_k, max_k, gen, num_gen):
    """[A] Tournament size tăng dần từ base_k → max_k theo tiến trình thế hệ."""
    progress = gen / max(num_gen, 1)
    k = int(round(base_k + (max_k - base_k) * progress))
    return max(2, min(k, max_k))


# ============================================================
# [E][I] NICHE-BASED EXTERNAL PARETO ARCHIVE
# ============================================================

def _filter_non_dominated_feasible(candidates):
    """Lọc tập non-dominated từ danh sách candidates feasible."""
    non_dom = []
    for ind in candidates:
        dominated = False
        for other in candidates:
            if other is ind:
                continue
            if _dominates(other.objectives, ind.objectives):
                dominated = True
                break
        if not dominated:
            non_dom.append(ind)
    return non_dom


def update_archive(archive, new_candidates, max_size,
                   n_niches=10, epsilon=ARCHIVE_EPSILON):
    """
    [E][I][L] Niche-based archive với epsilon-dominance.

    Sửa lỗi archive collapse:
      - Niche key dùng vector (F1_bucket, F2_bucket, F3_bucket) thay vì sum
      - Min-archive guard: nếu |eps_non_dom| < 5 → fallback về non_dom thường
      - epsilon nhỏ hơn (0.02) để chỉ loại nghiệm thực sự trùng lắp
    """
    feasible = [ind for ind in new_candidates if getattr(ind, "is_feasible", False)]
    if not feasible and not archive:
        return archive

    combined = archive + feasible

    # Bước 1: lọc non-dominated thông thường
    non_dom = _filter_non_dominated_feasible(combined)

    # Bước 2: [I] Epsilon-dominance — chỉ áp dụng khi đủ nghiệm
    if len(non_dom) >= 5:
        eps_non_dom = []
        for ind in non_dom:
            eps_dominated = False
            for other in non_dom:
                if other is ind:
                    continue
                if _epsilon_dominates(other.objectives, ind.objectives, eps=epsilon):
                    eps_dominated = True
                    break
            if not eps_dominated:
                eps_non_dom.append(ind)
        # Guard: epsilon không được xóa quá nhiều nghiệm
        unique = eps_non_dom if len(eps_non_dom) >= max(3, len(non_dom) // 4) else non_dom
    else:
        unique = non_dom  # Không đủ nghiệm → bỏ qua epsilon-dominance

    # Loại trùng lặp objectives chính xác
    seen_obj = set()
    deduped = []
    for ind in unique:
        if ind.objectives not in seen_obj:
            seen_obj.add(ind.objectives)
            deduped.append(ind)

    if len(deduped) <= max_size:
        return deduped

    # Bước 3: [E] Niche-based cắt bớt — dùng vector (F1, F2, F3) riêng lẻ
    # QUAN TRỌNG: Không dùng sum() vì F1=100,F2=50 và F1=50,F2=100 có cùng sum
    # nhưng đại diện cho 2 trade-off hoàn toàn khác nhau trên Pareto front.
    obj_mins = [min(ind.objectives[i] for ind in deduped) for i in range(3)]
    obj_maxs = [max(ind.objectives[i] for ind in deduped) for i in range(3)]
    obj_spans = [max(mx - mn, 1e-9) for mn, mx in zip(obj_mins, obj_maxs)]

    niches = defaultdict(list)
    for ind in deduped:
        # Tạo niche ID là tuple 3D — giữ nguyên trade-off structure
        niche_key = tuple(
            int((ind.objectives[i] - obj_mins[i]) / obj_spans[i] * (n_niches - 1))
            for i in range(3)
        )
        niches[niche_key].append(ind)

    crowding_distance_assignment(deduped)

    selected = []
    niche_queues = {}
    for nid, inds in niches.items():
        niche_queues[nid] = sorted(inds, key=lambda x: -getattr(x, "crowding_distance", 0.0))

    # Ưu tiên niche ít nghiệm trước (diversity-preserving)
    niche_order = sorted(niche_queues.keys(), key=lambda k: len(niche_queues[k]))

    round_idx = 0
    while len(selected) < max_size:
        added_any = False
        for nid in niche_order:
            if round_idx < len(niche_queues[nid]):
                selected.append(niche_queues[nid][round_idx])
                if len(selected) >= max_size:
                    break
                added_any = True
        if not added_any:
            break
        round_idx += 1

    return selected[:max_size]


# ============================================================
# DIVERSITY UTILITIES
# ============================================================

def population_diversity(population):
    """[F] Diversity dùng signature đầy đủ: (home, away, round, timeslot)."""
    sigs = set()
    for ind in population:
        sig = tuple(sorted(
            (m.home_team_id, m.away_team_id, m.round, m.timeslot_id)
            for m in ind.matches
        ))
        sigs.add(sig)
    return len(sigs) / len(population) if population else 0.0


def _make_new_individual(tournament):
    """Tạo một cá thể mới từ đầu + repair + objectives."""
    fn  = random.choice([_create_round_robin_individual, _create_biased_individual])
    ind = fn(tournament)
    repair_individual(ind, tournament)
    repair_no_consec_big_team(ind, tournament)
    repair_big_match_alternation(ind, tournament)
    compute_objectives(ind, tournament)
    return ind


def inject_diversity(population, tournament, inject_count, aggressive=True):
    """
    Thay thế inject_count cá thể kém nhất:
      60% → đột biến mạnh từ rank-0 (chọn ngẫu nhiên [J])
      40% → cá thể hoàn toàn mới
    """
    population.sort(key=lambda ind: (ind.rank, -getattr(ind, "crowding_distance", 0.0)))
    keep  = population[:-inject_count]
    rank0 = [ind for ind in population if ind.rank == 0][:8]
    new_inds = []

    for _ in range(inject_count):
        if rank0 and random.random() < 0.60:
            base = copy.deepcopy(random.choice(rank0))
            base = mutate(base, tournament, 0.80, aggressive=aggressive)
            repair_individual(base, tournament)
            repair_no_consec_big_team(base, tournament)
            repair_big_match_alternation(base, tournament)
            compute_objectives(base, tournament)
        else:
            base = _make_new_individual(tournament)
        new_inds.append(base)

    return keep + new_inds


def partial_restart(population, tournament, n_new, elitism_count, archive=None):
    """
    Giữ lại elitism_count cá thể tốt nhất (feasible rank-0),
    tạo mới n_new cá thể.

    [J] Seed từ archive theo random sample, không lấy archive[:k] cố định.
    [L] Ưu tiên giữ nghiệm đa dạng từ archive vì Pareto front rộng hơn.
    """
    rank0     = [ind for ind in population if getattr(ind, "rank", 999) == 0
                 and getattr(ind, "is_feasible", False)]
    survivors = sorted(rank0, key=lambda ind: -getattr(ind, "crowding_distance", 0.0))[:elitism_count]

    new_inds = []
    for _ in range(n_new):
        ind = _create_round_robin_individual(tournament)
        repair_individual(ind, tournament)
        local_search_big_match_chain(ind, tournament, max_iter=15)
        compute_objectives(ind, tournament)
        new_inds.append(ind)

    # [J] Seed từ archive NGẪU NHIÊN
    if archive:
        seed_count = min(elitism_count, len(archive))
        archive_seeds = random.sample(archive, seed_count)
        archive_copies = [copy.deepcopy(ind) for ind in archive_seeds]
        if len(new_inds) >= seed_count:
            new_inds[-seed_count:] = archive_copies
        else:
            new_inds = archive_copies + new_inds

    rest = [ind for ind in population if ind not in survivors]
    rest.sort(key=lambda ind: (ind.rank, -getattr(ind, "crowding_distance", 0.0)))
    filler = max(0, len(population) - len(survivors) - len(new_inds))
    return survivors + new_inds + rest[:filler]


# ============================================================
# HYPERVOLUME (Monte Carlo) — [B]
# ============================================================

def _hypervolume_mc(front, ref_point, n_samples=2000):
    """Ước lượng hypervolume của Pareto front bằng Monte Carlo."""
    feasible = [ind for ind in front if getattr(ind, "is_feasible", False)]
    if not feasible:
        return 0.0

    num_obj    = len(feasible[0].objectives)
    obj_matrix = [ind.objectives for ind in feasible]
    lb = [min(o[m] for o in obj_matrix) for m in range(num_obj)]
    ub = list(ref_point)

    box_vol = 1.0
    for lo, hi in zip(lb, ub):
        if hi <= lo:
            return 0.0
        box_vol *= (hi - lo)

    hits = 0
    for _ in range(n_samples):
        pt = tuple(random.uniform(lb[m], ub[m]) for m in range(num_obj))
        if any(all(o[m] <= pt[m] for m in range(num_obj)) for o in obj_matrix):
            hits += 1

    return box_vol * hits / n_samples


def _build_ref_point(fronts, margin=0.5):
    """Xây reference point từ worst objectives trong quần thể."""
    all_inds = [ind for f in fronts for ind in f if getattr(ind, "is_feasible", False)]
    if not all_inds:
        return (50_000.0, 50_000.0, 50_000.0)
    num_obj = len(all_inds[0].objectives)
    return tuple(
        max(ind.objectives[m] for ind in all_inds) * (1 + margin) + 1.0
        for m in range(num_obj)
    )


# ============================================================
# CHỌN CÁ THỂ ĐẠI DIỆN TỪ PARETO
# ============================================================

def pick_representative(pareto, strategy="balanced"):
    """
    Chọn một cá thể đại diện từ Pareto front.

    strategy:
      "balanced"  — tổng F1+F2+F3 nhỏ nhất (mặc định)
      "fairness"  — F1 (Công bằng đội) nhỏ nhất
      "quality"   — F2 (Chất lượng trận) nhỏ nhất
      "broadcast" — F3 (Vận hành/Phát sóng) nhỏ nhất
      "knee"      — điểm knee (xa nhất so với đường thẳng hai đầu)

    [L] "ops" → đổi tên thành "quality" cho phù hợp với F2 mới.
        Backward compat: "ops" vẫn hoạt động → map về "quality".
    """
    if not pareto:
        return None

    # Backward compat
    if strategy == "ops":
        strategy = "quality"

    if strategy == "fairness":
        return min(pareto, key=lambda ind: ind.objectives[0])
    if strategy == "quality":
        return min(pareto, key=lambda ind: ind.objectives[1])
    if strategy == "broadcast":
        return min(pareto, key=lambda ind: ind.objectives[2])
    if strategy == "balanced":
        return min(pareto, key=lambda ind: sum(ind.objectives))
    if strategy == "knee":
        sorted_p = sorted(pareto, key=lambda ind: ind.objectives[0])
        if len(sorted_p) < 3:
            return sorted_p[0]
        n_obj = len(sorted_p[0].objectives)
        mins  = [min(ind.objectives[i] for ind in sorted_p) for i in range(n_obj)]
        maxs  = [max(ind.objectives[i] for ind in sorted_p) for i in range(n_obj)]
        spans = [max(mx - mn, 1e-9) for mn, mx in zip(mins, maxs)]

        def norm(ind):
            return tuple((ind.objectives[i] - mins[i]) / spans[i] for i in range(n_obj))

        a = norm(sorted_p[0])
        b = norm(sorted_p[-1])
        ab = tuple(b[i] - a[i] for i in range(n_obj))
        ab_len = math.sqrt(sum(v ** 2 for v in ab)) or 1e-9

        def dist_to_line(ind):
            p  = norm(ind)
            ap = tuple(p[i] - a[i] for i in range(n_obj))
            dot = sum(ap[i] * ab[i] for i in range(n_obj))
            proj = tuple(dot / (ab_len ** 2) * ab[i] for i in range(n_obj))
            perp = tuple(ap[i] - proj[i] for i in range(n_obj))
            return math.sqrt(sum(v ** 2 for v in perp))

        return max(sorted_p, key=dist_to_line)

    return min(pareto, key=lambda ind: sum(ind.objectives))


# ============================================================
# VÒNG LẶP CHÍNH NSGA-II
# ============================================================

def run_nsga2(tournament, params=None):
    """
    Chạy NSGA-II đa mục tiêu với Pareto front đa dạng hóa.

    [L] Sau khi tách objectives (Hướng 4), kỳ vọng:
      - Pareto front rộng hơn (nhiều nghiệm hơn)
      - HV tăng chậm hơn nhưng spread lớn hơn
      - archive_size thường đạt max do front thật sự đa dạng

    Parameters
    ----------
    tournament : Tournament
    params     : dict — tham số tuỳ chỉnh (ghi đè auto-scale). None = dùng auto-scale.

    Returns
    -------
    pareto_front : list[Individual]   — Pareto rank-0 feasible cuối cùng (từ archive)
    history      : list[dict]         — Log theo từng thế hệ
    """
    resolved = build_params(tournament.num_teams, overrides=params)
    start_time = time.time()

    pop_size        = resolved["population_size"]
    num_gen         = resolved["num_generations"]
    cx_rate         = resolved["crossover_rate"]
    base_mut        = resolved["mutation_rate"]
    mut_max         = resolved["mut_rate_max"]
    tourn_k_base    = resolved["tournament_size"]
    tourn_k_max     = resolved["tournament_size_max"]
    stag_pat        = resolved["stagnation_patience"]
    stag_tol        = resolved["stagnation_tol"]
    hv_window       = resolved["hv_window"]
    agg_after       = resolved["aggressive_after"]
    inject_ratio    = resolved["inject_ratio"]
    bloat_thresh    = resolved["front0_bloat_thresh"]
    ls_interval     = resolved["local_search_interval"]
    ls_topk         = resolved["local_search_top_k"]
    restart_lim     = resolved["restart_limit"]
    div_thresh      = resolved["diversity_threshold"]
    multi_prob      = resolved["multi_mutation_prob"]
    ls_its          = resolved["ls_max_iter_timeslot"]
    ls_irr          = resolved["ls_max_iter_round"]
    ls_ibc          = resolved["ls_max_iter_chain"]
    archive_max     = resolved["archive_max_size"]
    archive_niches  = resolved["archive_n_niches"]
    archive_eps     = resolved["archive_epsilon"]
    oversample_f    = resolved["oversample_factor"]
    oversample_aft  = resolved["oversample_after"]
    rank_ratio      = resolved["rank_select_ratio"]
    ref_divisions   = resolved["ref_point_divisions"]

    inject_count = max(1, int(pop_size * inject_ratio))

    # [K] Tạo reference points một lần
    ref_points = generate_reference_points(n_obj=3, n_divisions=ref_divisions)

    def _fmin(f0, idx):
        vals = [ind.objectives[idx] for ind in f0 if getattr(ind, "is_feasible", False)]
        return min(vals) if vals else float('inf')

    # ── 1. Khởi tạo quần thể ──────────────────────────────────
    print(f"\n{'='*72}")
    print(f"[NSGA-II][L] Giải: {getattr(tournament, 'name', '?')} | "
          f"{tournament.num_teams} đội | {tournament.num_rounds} vòng | "
          f"{tournament.num_teams * (tournament.num_teams - 1)} trận")
    print(f"[NSGA-II][L] Phân nhóm objectives (Hướng 4 — conflict thật):")
    print(f"  F1 Công bằng đội    : {', '.join(KEYS_F1)}")
    print(f"  F2 Chất lượng trận  : {', '.join(KEYS_F2)}")
    print(f"  F3 Vận hành/Phát sóng: {', '.join(KEYS_F3)}")
    print(f"[NSGA-II] Tham số:")
    print(f"  pop={pop_size}  gen={num_gen}  cx={cx_rate}  "
          f"mut={base_mut}~{mut_max}  tourn={tourn_k_base}~{tourn_k_max}")
    print(f"  inject={inject_count}  ls_topk={ls_topk}  stag_pat={stag_pat}")
    print(f"  archive_max={archive_max}  archive_niches={archive_niches}  "
          f"archive_eps={archive_eps}")
    print(f"  ref_points={len(ref_points)}  hv_window={hv_window}  "
          f"oversample={oversample_f}x (sau {oversample_aft} stag)  rank_ratio={rank_ratio}")
    print(f"{'='*72}\n")

    print(f"[NSGA-II] Khởi tạo quần thể ({pop_size} cá thể)...")
    population = create_population(tournament, pop_size)
    for ind in population:
        repair_individual(ind, tournament)
        repair_no_consec_big_team(ind, tournament)
        repair_big_match_alternation(ind, tournament)
        compute_objectives(ind, tournament)

    fronts = fast_non_dominated_sort(population)
    for front in fronts:
        crowding_distance_assignment(front)

    ref_point = _build_ref_point(fronts)
    hv        = _hypervolume_mc(fronts[0], ref_point)

    # [E] Khởi tạo external archive với niche-based
    archive = update_archive([], fronts[0], archive_max,
                             n_niches=archive_niches, epsilon=archive_eps)

    # [D] Trung bình trượt HV
    hv_history_window = deque([hv] * hv_window, maxlen=hv_window)
    best_hv_avg = hv

    stag     = 0
    mut_rate = base_mut
    history  = []

    # [K] Niche count ban đầu
    niche_count = associate_to_reference_points(
        [ind for ind in population if getattr(ind, "is_feasible", False)],
        ref_points
    )

    feasible_init = sum(1 for ind in population if getattr(ind, "is_feasible", False))
    print(f"[NSGA-II] Khởi tạo xong — feasible: {feasible_init}/{pop_size} | "
          f"Front-0: {len(fronts[0])} cá thể | Archive: {len(archive)} | "
          f"F1={_fmin(fronts[0],0):.0f}  F2={_fmin(fronts[0],1):.0f}  "
          f"F3={_fmin(fronts[0],2):.0f} | HV={hv:.4f}")
    print(f"[NSGA-II] Ref points: {len(ref_points)} | "
          f"Bắt đầu tiến hóa ({num_gen} thế hệ)...\n")

    for gen in range(1, num_gen + 1):

        # ── [A] Adaptive tournament_size ──────────────────────
        tourn_k = _adaptive_tournament_k(tourn_k_base, tourn_k_max, gen, num_gen)

        # ── Adaptive mutation rate ─────────────────────────────
        if stag > 0:
            ramp     = stag / max(stag_pat, 1)
            mut_rate = min(mut_max, base_mut * (1.0 + 2.5 * ramp))
        else:
            mut_rate = base_mut
        aggressive = stag >= agg_after

        # ── [K] Cập nhật niche count mỗi 5 thế hệ ────────────
        if gen % 5 == 0:
            feasible_pop = [ind for ind in population if getattr(ind, "is_feasible", False)]
            niche_count = associate_to_reference_points(feasible_pop, ref_points)

        # ── Local search định kỳ trên top Pareto ──────────────
        if gen % ls_interval == 0:
            top_inds = sorted(
                [ind for ind in fronts[0] if getattr(ind, "is_feasible", False)],
                key=lambda ind: -getattr(ind, "crowding_distance", 0.0)
            )[:ls_topk]
            for ind in top_inds:
                local_search_timeslot(ind, tournament, max_iter=ls_its)
                local_search_round_swap(ind, tournament, max_iter=ls_irr)
                local_search_big_match_chain(ind, tournament, max_iter=ls_ibc)
                compute_objectives(ind, tournament)
            archive = update_archive(archive, top_inds, archive_max,
                                     n_niches=archive_niches, epsilon=archive_eps)

        # ── [J] Partial restart khi stagnation kéo dài ────────
        if stag > 0 and stag % restart_lim == 0:
            n_new     = pop_size // 3
            elitism_k = max(4, pop_size // 6)
            population = partial_restart(population, tournament, n_new, elitism_k,
                                         archive=archive)
            fronts = fast_non_dominated_sort(population)
            for front in fronts:
                crowding_distance_assignment(front)
            if gen % HV_INTERVAL == 0:
                hv = _hypervolume_mc(fronts[0], ref_point)
                hv_history_window.append(hv)
            print(f"  [Gen {gen:4d}] PARTIAL RESTART (stagnation={stag}) | Archive: {len(archive)}")
            stag = 0
            continue

        # ── Diversity injection khi diversity thấp ────────────
        div = population_diversity(population)
        if div < div_thresh and stag > 5:
            population = inject_diversity(population, tournament, inject_count, aggressive)
            fronts = fast_non_dominated_sort(population)
            for front in fronts:
                crowding_distance_assignment(front)
            if gen % HV_INTERVAL == 0:
                hv = _hypervolume_mc(fronts[0], ref_point)
                hv_history_window.append(hv)
            archive = update_archive(archive, fronts[0], archive_max,
                                     n_niches=archive_niches, epsilon=archive_eps)
            stag = 0
            print(f"  [Gen {gen:4d}] DIVERSITY INJECT (div={div:.2f}) | Archive: {len(archive)}")
            continue

        # ── Bloat control: front-0 quá lớn ───────────────────
        if len(fronts[0]) > pop_size * bloat_thresh and stag >= 4:
            population = inject_diversity(population, tournament, inject_count, aggressive)
            fronts = fast_non_dominated_sort(population)
            for front in fronts:
                crowding_distance_assignment(front)

        # ── [G] Xác định offspring_size (oversampling) ────────
        if stag >= oversample_aft:
            offspring_size = int(pop_size * oversample_f)
        else:
            offspring_size = pop_size

        # ── Tạo offspring ──────────────────────────────────────
        offspring = []
        while len(offspring) < offspring_size:
            p1 = _select_parent(population, tourn_k, rank_ratio, niche_count)
            p2 = _select_parent(population, tourn_k, rank_ratio, niche_count)

            if random.random() < cx_rate:
                c1, c2 = crossover(p1, p2, tournament)
            else:
                c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)

            c1 = mutate(c1, tournament, mut_rate, aggressive=aggressive)
            c2 = mutate(c2, tournament, mut_rate, aggressive=aggressive)

            if random.random() < multi_prob:
                extra = random.choice([
                    shuffle_round_mutation,
                    swap_team_identities_mutation,
                    cyclic_shift_mutation,
                ])
                c1 = extra(c1, tournament)
                c2 = extra(c2, tournament)

            for child in (c1, c2):
                repair_individual(child, tournament)
                repair_no_consec_big_team(child, tournament)
                repair_big_match_alternation(child, tournament)
                compute_objectives(child, tournament)

            remaining = offspring_size - len(offspring)
            offspring.extend([c1, c2][:remaining])

        # ── NSGA-II environmental selection ───────────────────
        archive_copies = [copy.deepcopy(ind) for ind in archive]
        combined = population + offspring + archive_copies

        fronts   = fast_non_dominated_sort(combined)
        for front in fronts:
            crowding_distance_assignment(front)

        new_pop = []
        for front in fronts:
            if len(new_pop) + len(front) <= pop_size:
                new_pop.extend(front)
            else:
                remaining = pop_size - len(new_pop)
                front_sorted = sorted(
                    front,
                    key=lambda ind: (
                        -int(getattr(ind, "is_feasible", False)),
                        niche_count.get(getattr(ind, "_ref_point", -1), 999),
                        -getattr(ind, "crowding_distance", 0.0)
                    )
                )
                new_pop.extend(front_sorted[:remaining])
                break

        population = new_pop
        W = _get_weights(tournament.num_teams)
        feasibles = [ind for ind in population if ind.is_feasible][:50]

        for label, keys in [("F1", KEYS_F1), ("F2", KEYS_F2), ("F3", KEYS_F3)]:
            vals = []
            for ind in feasibles:
                soft = count_soft_penalties(ind, tournament)
                vals.append(sum(W.get(k,1) * soft.get(k,0) for k in keys))
            print(f"{label}: min={min(vals):.1f} max={max(vals):.1f} "
                f"mean={sum(vals)/len(vals):.1f} nonzero={sum(1 for v in vals if v>0)}/{len(vals)}")
            # In chi tiết từng key
            for k in keys:
                kvals = [W.get(k,1) * count_soft_penalties(ind, tournament).get(k,0) 
                        for ind in feasibles]
                print(f"  {k}: mean={sum(kvals)/len(kvals):.2f} max={max(kvals):.2f}")

        fronts = fast_non_dominated_sort(population)
        for front in fronts:
            crowding_distance_assignment(front)

        # [E] Cập nhật niche-based archive sau mỗi thế hệ
        archive = update_archive(archive, fronts[0], archive_max,
                                 n_niches=archive_niches, epsilon=archive_eps)

        # ── [B][D] Cập nhật hypervolume & stagnation ──────────
        if gen % HV_INTERVAL == 0:
            hv = _hypervolume_mc(fronts[0], ref_point)
            hv_history_window.append(hv)

        hv_avg = sum(hv_history_window) / len(hv_history_window)
        if (hv_avg - best_hv_avg) / (abs(best_hv_avg) + 1e-12) > stag_tol:
            best_hv_avg = hv_avg
            stag        = 0
        else:
            stag += 1

        # ── Ghi history ───────────────────────────────────────
        f0     = fronts[0]
        f1_min = _fmin(f0, 0)
        f2_min = _fmin(f0, 1)
        f3_min = _fmin(f0, 2)
        history.append({
            "gen"         : gen,
            "front0_size" : len(f0),
            "f1_min"      : f1_min,
            "f2_min"      : f2_min,
            "f3_min"      : f3_min,
            "hv"          : hv,
            "hv_avg"      : hv_avg,
            "mut_rate"    : mut_rate,
            "diversity"   : div,
            "stagnation"  : stag,
            "feasible_f0" : sum(1 for ind in f0 if getattr(ind, "is_feasible", False)),
            "archive_size": len(archive),
            "tourn_k"     : tourn_k,
            "offspring_sz": offspring_size,
        })

        # ── Log mỗi 5 thế hệ ──────────────────────────────────
        if gen % 5 == 0 or stag >= stag_pat - 5:
            feasible_cnt = sum(1 for ind in f0 if getattr(ind, "is_feasible", False))
            mode = " [AGG]" if aggressive else ""
            if len(archive) >= 2:
                f1_vals = [ind.objectives[0] for ind in archive]
                f2_vals = [ind.objectives[1] for ind in archive]
                f3_vals = [ind.objectives[2] for ind in archive]
                spread = (max(f1_vals) - min(f1_vals) +
                          max(f2_vals) - min(f2_vals) +
                          max(f3_vals) - min(f3_vals))
            else:
                spread = 0.0
            print(f"Gen {gen:4d} | F0: {len(f0):3d} ({feasible_cnt:3d} feas) | "
                  f"F1={f1_min:6.0f}  F2={f2_min:5.0f}  F3={f3_min:5.0f} | "
                  f"HV={hv:.4f}(avg={hv_avg:.4f}) | μ={mut_rate:.3f} | "
                  f"k={tourn_k} | div={div:.2f} | arch={len(archive)} | "
                  f"spread={spread:.0f} | stag={stag}{mode}")

        # ── Early stopping ─────────────────────────────────────
        if stag >= stag_pat:
            print(f"\n[NSGA-II] Early stopping tại gen {gen} "
                  f"(HV avg không cải thiện {stag_pat} thế hệ liên tiếp)")
            break

    # ── Kết quả cuối — lấy từ ARCHIVE ─────────────────────────
    pareto = sorted(
        [ind for ind in archive if getattr(ind, "is_feasible", False)],
        key=lambda ind: ind.objectives,
    )

    if not pareto:
        pareto = sorted(
            [ind for ind in fronts[0] if getattr(ind, "is_feasible", False)],
            key=lambda ind: ind.objectives,
        )

    print(f"\n[NSGA-II] Hoàn tất!")
    print(f"  Pareto front (từ archive, feasible): {len(pareto)} cá thể")

    if len(pareto) >= 2:
        f1v = [ind.objectives[0] for ind in pareto]
        f2v = [ind.objectives[1] for ind in pareto]
        f3v = [ind.objectives[2] for ind in pareto]
        print(f"  F1 Công bằng đội     : [{min(f1v):.0f}, {max(f1v):.0f}] "
              f"(spread={max(f1v)-min(f1v):.0f})")
        print(f"  F2 Chất lượng trận   : [{min(f2v):.0f}, {max(f2v):.0f}] "
              f"(spread={max(f2v)-min(f2v):.0f})")
        print(f"  F3 Vận hành/Phát sóng: [{min(f3v):.0f}, {max(f3v):.0f}] "
              f"(spread={max(f3v)-min(f3v):.0f})")

    duration = time.time() - start_time
    print(f"  Thời gian thực hiện: {duration:.2f} giây")

    return pareto, history, duration

# ============================================================
# ENTRY POINT / TEST
# ============================================================

if __name__ == "__main__":
    import os
    import sys
    import json
    from datetime import datetime
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from models import Tournament
    from constraints import count_hard_violations, count_soft_penalties
    from ga import _create_round_robin_individual, repair_individual

    base_dir   = os.path.dirname(os.path.abspath(__file__))
    data_file  = os.path.join(base_dir, "../data/teams8.json")
    tournament = Tournament(data_file)

    print("\n--- KIỂM TRA TOURNAMENT ---")
    print(f"num_teams    : {tournament.num_teams}")
    print(f"num_rounds   : {tournament.num_rounds}")
    print(f"num_timeslots: {tournament.num_timeslots}")
    print(f"start_date   : {tournament.start_date.date()}")
    print(f"end_date     : {tournament.end_date.date()}")

    ind = _create_round_robin_individual(tournament)
    print(f"\nTrước repair: {len(ind.matches)} trận")
    hard_before = count_hard_violations(ind, tournament)
    for k, v in hard_before.items():
        print(f"  {'✓' if v==0 else '✗'} {k}: {v}")

    repair_individual(ind, tournament)
    print(f"\nSau repair:")
    hard_after = count_hard_violations(ind, tournament)
    for k, v in hard_after.items():
        print(f"  {'✓' if v==0 else '✗'} {k}: {v}")
    print(f"  Tổng vi phạm: {sum(hard_after.values())}")
    print("─" * 42)

    print("=" * 72)
    print("  F1 — Công bằng đội    : home/away balance, rest days, season edge")
    print("  F2 — Chất lượng trận  : derby, big match distribution & alternation")
    print("  F3 — Vận hành/Phát sóng: prime slots, slot overload, travel balance")
    print("=" * 72)

    OVERRIDES = {
        #   "num_generations": 100,  # bỏ comment để test nhanh
    }

    pareto, history, duration = run_nsga2(tournament, params=OVERRIDES or None)

    # ── In Pareto front ────────────────────────────────────────
    W = _get_weights(tournament.num_teams)
    print(f"\n{'─'*72}")
    print(f"  PARETO FRONT [Hướng 4] — {len(pareto)} cá thể tốt nhất (feasible, từ archive)")
    print(f"{'─'*72}")
    print(f"  {'#':>3}  {'F1 Công bằng':>14}  {'F2 Chất lượng':>14}  {'F3 Vận hành':>13}  {'Tổng':>8}")
    print(f"  {'─'*3}  {'─'*14}  {'─'*14}  {'─'*13}  {'─'*8}")
    for i, ind in enumerate(sorted(pareto, key=lambda x: x.objectives[0])[:20]):
        f1, f2, f3 = ind.objectives
        print(f"  {i+1:>3}  {f1:>14.1f}  {f2:>14.1f}  {f3:>13.1f}  {f1+f2+f3:>8.1f}")
    if len(pareto) > 20:
        print(f"  ... và {len(pareto) - 20} cá thể nữa")

    # ── Chi tiết các cá thể đại diện ──────────────────────────
    strategies = [
        ("balanced",  "Tổng F1+F2+F3 nhỏ nhất"),
        ("fairness",  "F1 (Công bằng đội) tốt nhất"),
        ("quality",   "F2 (Chất lượng trận) tốt nhất"),
        ("broadcast", "F3 (Vận hành/Phát sóng) tốt nhất"),
        ("knee",      "Điểm Knee (cân bằng)"),
    ]
    for strat, label in strategies:
        rep = pick_representative(pareto, strategy=strat)
        if rep is None:
            continue
        f1, f2, f3 = rep.objectives
        print(f"\n{'─'*72}")
        print(f"  [{label}]  F1={f1:.1f}  F2={f2:.1f}  F3={f3:.1f}  (Σ={f1+f2+f3:.1f})")

        hard = count_hard_violations(rep, tournament)
        hard_ok = all(v == 0 for v in hard.values())
        print(f"  Ràng buộc cứng: {'✓ Tất cả hợp lệ' if hard_ok else str(hard)}")

        soft = count_soft_penalties(rep, tournament)
        groups = {
            "F1 — Công bằng đội    ": KEYS_F1,
            "F2 — Chất lượng trận  ": KEYS_F2,
            "F3 — Vận hành/Phát sóng": KEYS_F3,
        }
        for gname, keys in groups.items():
            parts = [f"{k}={soft.get(k,0)}" for k in keys if soft.get(k, 0) > 0]
            if parts:
                print(f"  {gname}: {', '.join(parts)}")
            else:
                print(f"  {gname}: ✓")

    # ── Tiến trình HV ──────────────────────────────────────────
    if history:
        print(f"\n{'─'*72}")
        print("  TIẾN TRÌNH HYPERVOLUME + SPREAD")
        print(f"{'─'*72}")
        checkpoints = sorted({0, len(history)//4, len(history)//2,
                               3*len(history)//4, len(history)-1})
        for cp in checkpoints:
            h = history[cp]
            print(f"  Gen {h['gen']:4d} | HV={h['hv']:.4f}(avg={h['hv_avg']:.4f}) | "
                  f"F1={h['f1_min']:.0f}  F2={h['f2_min']:.0f}  F3={h['f3_min']:.0f} | "
                  f"arch={h.get('archive_size','?')}  k={h.get('tourn_k','?')}")

    # ── LƯU DỮ LIỆU VẼ ĐỒ THỊ ────────────────────────────────
    result_dir = os.path.join(base_dir, "../results")
    os.makedirs(result_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    n  = tournament.num_teams

    # ── 1. Pareto front (Hình 4.3 — scatter F1 vs F2 vs F3) ───
    pareto_data = [
        {
            "f1": ind.objectives[0],
            "f2": ind.objectives[1],
            "f3": ind.objectives[2],
            "total": sum(ind.objectives)
        }
        for ind in pareto
    ]
    path_pareto = os.path.join(result_dir, f"pareto_{n}teams_{ts}.json")
    with open(path_pareto, "w", encoding="utf-8") as f:
        json.dump(pareto_data, f, indent=2)
    print(f"✅ Pareto front ({len(pareto)} điểm) → {path_pareto}")

    # ── 2. HV theo thế hệ (Hình 4.4 — convergence curve) ──────
    # Lọc chỉ lấy các gen có ghi HV (mỗi HV_INTERVAL thế hệ)
    hv_curve = [
        {"gen": h["gen"], "hv": h["hv"], "hv_avg": h["hv_avg"]}
        for h in history
        if h["hv"] > 0
    ]
    path_hv = os.path.join(result_dir, f"hv_curve_{n}teams_{ts}.json")
    with open(path_hv, "w", encoding="utf-8") as f:
        json.dump(hv_curve, f, indent=2)
    print(f"✅ HV curve ({len(hv_curve)} điểm) → {path_hv}")

    # ── 3. Pareto size theo thế hệ (Hình 4.4 — pareto size) ───
    pareto_size_curve = [
        {"gen": h["gen"], "pareto_size": h["front0_size"], "archive_size": h["archive_size"]}
        for h in history
    ]
    path_psize = os.path.join(result_dir, f"pareto_size_{n}teams_{ts}.json")
    with open(path_psize, "w", encoding="utf-8") as f:
        json.dump(pareto_size_curve, f, indent=2)
    print(f"✅ Pareto size curve ({len(pareto_size_curve)} điểm) → {path_psize}")

    # ── 4. Tóm tắt chỉ số cho Bảng 4.3 ───────────────────────
    # Tính HV normalized (chia cho ref_point volume để về [0,1])
    ref_pt = _build_ref_point([pareto], margin=0.5)
    hv_raw = _hypervolume_mc(pareto, ref_pt, n_samples=5000)
    ref_vol = 1.0
    for v in ref_pt:
        ref_vol *= v
    hv_normalized = hv_raw / ref_vol if ref_vol > 0 else 0.0

    # Tính IGD đơn giản: trung bình khoảng cách từ mỗi điểm archive
    # đến điểm gần nhất trong pareto (normalized)
    f1v = [ind.objectives[0] for ind in pareto]
    f2v = [ind.objectives[1] for ind in pareto]
    f3v = [ind.objectives[2] for ind in pareto]
    spans = [
        max(max(f1v) - min(f1v), 1e-9),
        max(max(f2v) - min(f2v), 1e-9),
        max(max(f3v) - min(f3v), 1e-9),
    ]

    def _norm_obj(obj):
        return tuple((obj[i] - [min(f1v), min(f2v), min(f3v)][i]) / spans[i] for i in range(3))

    def _dist(a, b):
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))

    pareto_norm = [_norm_obj(ind.objectives) for ind in pareto]
    if len(pareto_norm) >= 2:
        igd_vals = []
        for ref in pareto_norm:          # dùng chính pareto làm reference set
            d = min(_dist(ref, p) for p in pareto_norm if p != ref)
            igd_vals.append(d)
        igd = sum(igd_vals) / len(igd_vals)
    else:
        igd = 0.0

    # Thế hệ hội tụ: gen đầu tiên mà archive_size không tăng thêm trong 10 gen liên tiếp
    converge_gen = history[-1]["gen"]   # mặc định = gen cuối
    for i in range(10, len(history)):
        window = [history[j]["archive_size"] for j in range(i - 10, i)]
        if max(window) - min(window) <= 1:
            converge_gen = history[i - 10]["gen"]
            break
    def _rep_obj(strategy):
        rep = pick_representative(pareto, strategy=strategy)
        if rep is None:
            return None
        f1, f2, f3 = rep.objectives
        return {"f1": round(f1,1), "f2": round(f2,1), "f3": round(f3,1),
                "total": round(f1+f2+f3, 1)}
    summary = {
        "num_teams"       : n,
        "pareto_size"     : len(pareto),
        "hv_raw"          : round(hv_raw, 2),
        "hv_normalized"   : round(hv_normalized, 4),
        "igd"             : round(igd, 4),
        "converge_gen"    : converge_gen,
        "total_gen_run"   : history[-1]["gen"] if history else 0,
        "f1_range"        : [round(min(f1v), 1), round(max(f1v), 1)],
        "f2_range"        : [round(min(f2v), 1), round(max(f2v), 1)],
        "f3_range"        : [round(min(f3v), 1), round(max(f3v), 1)],
        "best_balanced"   : _rep_obj("balanced"),        # ← MỚI
        "best_f1"         : _rep_obj("fairness"),        # ← MỚI
        "best_f2"         : _rep_obj("quality"),         # ← MỚI
        "best_f3"         : _rep_obj("broadcast"),       # ← MỚI
    }
    path_summary = os.path.join(result_dir, f"summary_{n}teams_{ts}.json")
    with open(path_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✅ Summary (Bảng 4.3) → {path_summary}")
    print(f"   pareto_size={summary['pareto_size']}  "
          f"HV={summary['hv_normalized']:.3f}  "
          f"IGD={summary['igd']:.3f}  "
          f"converge_gen={summary['converge_gen']}")
    # ── 5. Lưu lịch thi đấu của các cá thể đại diện ──────────
    strategies_save = [
        ("balanced",  "balanced"),
        ("fairness",  "best_f1"),
        ("quality",   "best_f2"),
        ("broadcast", "best_f3"),
        ("knee",      "knee"),
    ]
    for strat, label in strategies_save:
        rep = pick_representative(pareto, strategy=strat)
        if rep is None:
            continue
        f1, f2, f3 = rep.objectives
        schedule_data = {
            "strategy"  : label,
            "objectives": {
                "f1"   : round(f1, 1),
                "f2"   : round(f2, 1),
                "f3"   : round(f3, 1),
                "total": round(f1 + f2 + f3, 1),
            },
            "matches": [
                {
                    "round"       : m.round,
                    "home_team_id": m.home_team_id,
                    "away_team_id": m.away_team_id,
                    "timeslot_id" : m.timeslot_id,
                }
                for m in sorted(rep.matches, key=lambda m: (m.round, m.timeslot_id))
            ],
        }
        path_schedule = os.path.join(result_dir, f"schedule_{label}_{n}teams_{ts}.json")
        with open(path_schedule, "w", encoding="utf-8") as f:
            json.dump(schedule_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Lịch thi đấu [{label}] ({len(schedule_data['matches'])} trận) → {path_schedule}")