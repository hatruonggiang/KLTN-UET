import random
import time
import copy
from collections import defaultdict
from models import Match, Individual
from fitness import calculate_fitness
from constraints import count_hard_violations, count_soft_penalties


# ============================================================
# THAM SỐ
# ============================================================

DEFAULT_PARAMS = {
    "population_size":       120,
    "num_generations":       100,
    "crossover_rate":        0.88,
    "mutation_rate":         0.28,
    "tournament_size":       8,
    "elitism_count":         8,
    "stagnation_limit":      12,
    "restart_limit":         40,
    "local_search_interval": 4,
    "local_search_top_k":    12,
}

CROSSOVER_WEIGHTS = {
    "match_swap":            0.22,
    "round_block_full_swap": 0.15,
    "round_match_swap":      0.12,
    "structural_round":      0.10,
    "round_swap":            0.10,
    "random_match_set":      0.08,
    "adjusted_round_swap":   0.08,
    "team_based":            0.06,
    "round_block":           0.05,
    "single_match":          0.02,
    "timeslot":              0.02,
}

MUTATION_WEIGHTS_NORMAL = {
    "swap_two_rounds":      0.13,
    "targeted_fix":         0.12,
    "break_big_chain":      0.10,
    "match_exchange":       0.09,
    "round_reassign":       0.08,
    "shuffle_round":        0.07,
    "leg_swap":             0.06,
    "home_swap_same_round": 0.06,
    "pair_swap":            0.05,
    "swap_team_identities": 0.04,
    "block_reverse":        0.04,
    "rotate_rounds":        0.04,
    "timeslot":             0.03,
    "block_timeslot":       0.03,
    "home_away_swap":       0.03,
    "multi_point":          0.03,
    "cyclic_shift":         0.02,
}

MUTATION_WEIGHTS_AGGRESSIVE = {
    "break_big_chain":      0.22,
    "targeted_fix":         0.15,
    "swap_two_rounds":      0.12,
    "match_exchange":       0.10,
    "shuffle_round":        0.08,
    "round_reassign":       0.07,
    "leg_swap":             0.06,
    "home_swap_same_round": 0.05,
    "pair_swap":            0.05,
    "block_reverse":        0.04,
    "rotate_rounds":        0.04,
    "swap_team_identities": 0.04,
    "timeslot":             0.03,
    "block_timeslot":       0.03,
    "home_away_swap":       0.02,
}


# ============================================================
# HÀM HỖ TRỢ
# ============================================================

def is_derby(tournament, team_a, team_b):
    return tournament.get_team(team_a).city == tournament.get_team(team_b).city


def is_big_match(tournament, team_a, team_b, threshold=None):
    if threshold is None:
        threshold = 8 if tournament.num_teams <= 14 else 9
    return (tournament.get_champ_potential(team_a) >= threshold and
            tournament.get_champ_potential(team_b) >= threshold)


def get_prime_timeslots(tournament):
    return [4, 6]


def _build_team_round_map(matches):
    trm = defaultdict(lambda: defaultdict(list))
    for idx, m in enumerate(matches):
        for team in (m.home_team_id, m.away_team_id):
            trm[team][m.round].append(idx)
    return trm


# ============================================================
# KHỞI TẠO QUẦN THỂ
# ============================================================

def _circle_method_single_leg(n):
    """Circle Method: tạo n-1 vòng cho 1 leg, đảm bảo HC1 + HC2."""
    fixed = 0
    rotating = list(range(1, n))
    rounds = []
    for _ in range(n - 1):
        circle = [fixed] + rotating
        rounds.append([(circle[i], circle[n - 1 - i]) for i in range(n // 2)])
        rotating = [rotating[-1]] + rotating[:-1]
    return rounds


def _create_round_robin_individual(tournament):
    """Double Round Robin đảm bảo HC1 + HC2. Leg 2 đảo home/away."""
    n, num_ts = tournament.num_teams, len(tournament.timeslots)
    leg1 = _circle_method_single_leg(n)
    matches = []
    for r_idx, pairs in enumerate(leg1):
        for home, away in pairs:
            matches.append(Match(home, away, r_idx + 1, random.randint(0, num_ts - 1)))
    for r_idx, pairs in enumerate(leg1):
        for home, away in pairs:
            matches.append(Match(away, home, r_idx + 1 + (n - 1), random.randint(0, num_ts - 1)))
    return Individual(matches)


def _create_biased_individual(tournament):
    """Shuffle thứ tự đội trước circle method — tăng diversity quần thể."""
    n, num_ts = tournament.num_teams, len(tournament.timeslots)
    order = list(range(n))
    random.shuffle(order)
    leg1 = _circle_method_single_leg(n)
    matches = []
    for r_idx, pairs in enumerate(leg1):
        for hi, ai in pairs:
            matches.append(Match(order[hi], order[ai], r_idx + 1, random.randint(0, num_ts - 1)))
    for r_idx, pairs in enumerate(leg1):
        for hi, ai in pairs:
            matches.append(Match(order[ai], order[hi], r_idx + 1 + (n - 1), random.randint(0, num_ts - 1)))
    return Individual(matches)


def create_population(tournament, size):
    print(f"Khởi tạo quần thể ({size} cá thể) cho {tournament.num_teams} đội...")
    pop = []
    for i in range(size):
        pop.append(_create_round_robin_individual(tournament))
        if (i + 1) % 20 == 0:
            print(f"  [{i+1:4d}/{size}] Hoàn thành")
    print("Khởi tạo quần thể xong.\n")
    return pop


# ============================================================
# REPAIR
# ============================================================

def repair_individual(individual, tournament):
    matches  = individual.matches
    NR, NT   = tournament.num_rounds, tournament.num_teams
    MAX_ITER = NR * 15

    occ = defaultdict(set)
    for m in matches:
        occ[m.home_team_id].add(m.round)
        occ[m.away_team_id].add(m.round)

    for _ in range(MAX_ITER):
        # Sửa HC2: mỗi đội chỉ 1 trận/vòng
        seen = defaultdict(set)
        conflict = None
        for m in matches:
            if m.round in seen[m.home_team_id] or m.round in seen[m.away_team_id]:
                conflict = m
                break
            seen[m.home_team_id].add(m.round)
            seen[m.away_team_id].add(m.round)

        if conflict is not None:
            m, old_r = conflict, conflict.round
            h, a = m.home_team_id, m.away_team_id
            free_both = [r for r in range(1, NR + 1) if r not in occ[h] and r not in occ[a]]
            if free_both:
                new_r = random.choice(free_both)
            else:
                h_free = [r for r in range(1, NR + 1) if r not in occ[h] and r != old_r]
                if h_free:
                    new_r = random.choice(h_free)
                    blocker = next((mm for mm in matches
                                    if mm.round == new_r and
                                    (mm.home_team_id == a or mm.away_team_id == a)), None)
                    if blocker:
                        other = blocker.home_team_id if blocker.away_team_id == a else blocker.away_team_id
                        occ[a].discard(new_r); occ[other].discard(new_r)
                        blocker.round = old_r
                        occ[a].add(old_r); occ[other].add(old_r)
                    else:
                        new_r = random.randint(1, NR)
                else:
                    new_r = random.randint(1, NR)
            occ[h].discard(old_r); occ[a].discard(old_r)
            m.round = new_r
            occ[h].add(new_r); occ[a].add(new_r)
            continue

    return individual


def repair_no_consec_big_team(individual, tournament):
    matches, NR = individual.matches, tournament.num_rounds
    for _ in range(50):
        team_big = defaultdict(list)
        for idx, m in enumerate(matches):
            if is_big_match(tournament, m.home_team_id, m.away_team_id):
                for t in (m.home_team_id, m.away_team_id):
                    team_big[t].append((m.round, idx))
        improved = False
        for lst in team_big.values():
            lst.sort()
            for i in range(len(lst) - 1):
                if lst[i+1][0] == lst[i][0] + 1:
                    for tidx in (lst[i][1], lst[i+1][1]):
                        m = matches[tidx]
                        busy = ({m2.round for m2 in matches
                                 if m2.home_team_id == m.home_team_id or m2.away_team_id == m.home_team_id} |
                                {m2.round for m2 in matches
                                 if m2.home_team_id == m.away_team_id or m2.away_team_id == m.away_team_id})
                        free = [r for r in range(1, NR + 1) if r not in busy and r != m.round]
                        if free:
                            m.round = random.choice(free)
                            improved = True
                            break
                    if improved:
                        break
            if improved:
                break
        if not improved:
            break
    return individual


def repair_big_match_alternation(individual, tournament, max_per_slot=2):
    matches = individual.matches
    num_ts = len(tournament.timeslots)
    big_count = defaultdict(int)
    big_indices = []
    for i, m in enumerate(matches):
        if is_big_match(tournament, m.home_team_id, m.away_team_id):
            big_count[m.timeslot_id] += 1
            big_indices.append((i, m.timeslot_id))

    overloaded  = [ts for ts, cnt in big_count.items() if cnt > max_per_slot]
    underloaded = [ts for ts in range(num_ts) if big_count.get(ts, 0) < max_per_slot]
    if not overloaded or not underloaded:
        return individual

    for _ in range(5):
        ts_src = random.choice(overloaded)
        cands = [i for i, ts in big_indices if ts == ts_src]
        if not cands:
            break
        idx = random.choice(cands)
        ts_dst = random.choice(underloaded)
        matches[idx].timeslot_id = ts_dst
        big_count[ts_src] -= 1
        big_count[ts_dst] = big_count.get(ts_dst, 0) + 1
        if big_count[ts_src] <= max_per_slot and ts_src in overloaded:
            overloaded.remove(ts_src)
        if not overloaded or not underloaded:
            break
    return individual


# ============================================================
# LOCAL SEARCH
# ============================================================

def local_search_timeslot(individual, tournament, max_iter=30):
    from fitness import calculate_fitness as _calc
    num_ts = len(tournament.timeslots)
    matches = individual.matches
    cur_fit = individual.fitness or _calc(individual, tournament)
    for _ in range(max_iter):
        idx = random.randint(0, len(matches) - 1)
        old_ts = matches[idx].timeslot_id
        new_ts = random.randint(0, num_ts - 1)
        if new_ts == old_ts:
            continue
        matches[idx].timeslot_id = new_ts
        new_fit = _calc(individual, tournament)
        if new_fit > cur_fit:
            cur_fit = new_fit
        else:
            matches[idx].timeslot_id = old_ts
    individual.fitness = cur_fit
    return individual


def local_search_round_swap(individual, tournament, max_iter=20):
    from fitness import calculate_fitness as _calc
    matches = individual.matches
    NR = tournament.num_rounds
    cur_fit = individual.fitness
    trm = _build_team_round_map(matches)
    for _ in range(max_iter):
        idx = random.randint(0, len(matches) - 1)
        m, old_r = matches[idx], matches[idx].round
        busy_h = set(trm.get(m.home_team_id, {}).keys()) - {old_r}
        busy_a = set(trm.get(m.away_team_id, {}).keys()) - {old_r}
        free = [r for r in range(1, NR + 1) if r not in busy_h and r not in busy_a and r != old_r]
        if not free:
            continue
        matches[idx].round = random.choice(free)
        new_fit = _calc(individual, tournament)
        if new_fit >= cur_fit:
            cur_fit = new_fit
            trm = _build_team_round_map(matches)
        else:
            matches[idx].round = old_r
    individual.fitness = cur_fit
    return individual


def local_search_big_match_chain(individual, tournament, max_iter=25):
    from fitness import calculate_fitness as _calc
    matches = individual.matches
    NR = tournament.num_rounds
    cur_fit = individual.fitness
    if cur_fit is None:
        cur_fit = _calc(individual, tournament)

    for _ in range(max_iter):
        team_consec = defaultdict(list)
        for i, m in enumerate(matches):
            if is_big_match(tournament, m.home_team_id, m.away_team_id):
                for t in (m.home_team_id, m.away_team_id):
                    team_consec[t].append((m.round, i))

        improved = False
        for lst in team_consec.values():
            lst.sort()
            for k in range(len(lst) - 1):
                r1, i1 = lst[k]
                r2, i2 = lst[k + 1]
                if r2 != r1 + 1:
                    continue
                for tidx in (i1, i2):
                    m = matches[tidx]
                    old_r = m.round
                    busy = ({m2.round for m2 in matches
                             if m2.home_team_id == m.home_team_id or m2.away_team_id == m.home_team_id} |
                            {m2.round for m2 in matches
                             if m2.home_team_id == m.away_team_id or m2.away_team_id == m.away_team_id})
                    free = [r for r in range(1, NR + 1) if r not in busy and r != old_r]
                    best_r, best_fit = None, cur_fit
                    for nr in free:
                        m.round = nr
                        f = _calc(individual, tournament) or -9999999
                        if f > best_fit:
                            best_fit, best_r = f, nr
                        m.round = old_r
                    if best_r is not None:
                        m.round = best_r
                        cur_fit = best_fit
                        individual.fitness = cur_fit
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break

    individual.fitness = cur_fit
    return individual


# ============================================================
# SELECTION
# ============================================================

def tournament_selection(population, tournament_size):
    return max(random.sample(population, tournament_size), key=lambda ind: ind.fitness)


def rank_based_selection(population):
    sorted_pop = sorted(population, key=lambda ind: ind.fitness)
    weights = list(range(1, len(sorted_pop) + 1))
    return random.choices(sorted_pop, weights=weights, k=1)[0]


# ============================================================
# CROSSOVER
# ============================================================

def _p2_map(parent2):
    return {(m.home_team_id, m.away_team_id): m for m in parent2.matches}


def _p2_ts(parent2):
    return {(m.home_team_id, m.away_team_id): m.timeslot_id for m in parent2.matches}


def match_swap_crossover(p1, p2, tournament):
    mp2 = _p2_map(p2)
    c1, c2 = [], []
    for m in p1.matches:
        key = (m.home_team_id, m.away_team_id)
        p2m = mp2.get(key)
        if p2m and random.random() < 0.5:
            c1.append(Match(m.home_team_id, m.away_team_id, p2m.round, p2m.timeslot_id))
            c2.append(Match(m.home_team_id, m.away_team_id, m.round, m.timeslot_id))
        else:
            c1.append(Match(m.home_team_id, m.away_team_id, m.round, m.timeslot_id))
            c2.append(Match(m.home_team_id, m.away_team_id,
                            p2m.round if p2m else m.round,
                            p2m.timeslot_id if p2m else m.timeslot_id))
    c1i, c2i = Individual(c1), Individual(c2)
    repair_individual(c1i, tournament); repair_individual(c2i, tournament)
    return c1i, c2i


def round_block_full_swap_crossover(p1, p2, tournament):
    NR = tournament.num_rounds
    bs = random.randint(3, NR // 3)
    block = set(range(random.randint(1, NR - bs + 1), random.randint(1, NR - bs + 1) + bs))
    mp2 = {(m.home_team_id, m.away_team_id): m for m in p2.matches if m.round in block}
    mp1 = {(m.home_team_id, m.away_team_id): m for m in p1.matches if m.round in block}
    c1 = [Match(m.home_team_id, m.away_team_id,
                mp2[k].round if (k := (m.home_team_id, m.away_team_id)) in mp2 and m.round in block else m.round,
                mp2[k].timeslot_id if k in mp2 and m.round in block else m.timeslot_id)
          for m in p1.matches]
    c2 = [Match(m.home_team_id, m.away_team_id,
                mp1[k].round if (k := (m.home_team_id, m.away_team_id)) in mp1 and m.round in block else m.round,
                mp1[k].timeslot_id if k in mp1 and m.round in block else m.timeslot_id)
          for m in p2.matches]
    c1i, c2i = Individual(c1), Individual(c2)
    repair_individual(c1i, tournament); repair_individual(c2i, tournament)
    return c1i, c2i


def timeslot_crossover(p1, p2):
    ts2 = _p2_ts(p2)
    size = len(p1.matches)
    swap = set(random.sample(range(size), size // 2))
    c1, c2 = [], []
    for i, m in enumerate(p1.matches):
        key = (m.home_team_id, m.away_team_id)
        t2 = ts2.get(key, m.timeslot_id)
        if i in swap:
            c1.append(Match(m.home_team_id, m.away_team_id, m.round, t2))
            c2.append(Match(m.home_team_id, m.away_team_id, m.round, m.timeslot_id))
        else:
            c1.append(Match(m.home_team_id, m.away_team_id, m.round, m.timeslot_id))
            c2.append(Match(m.home_team_id, m.away_team_id, m.round, t2))
    return Individual(c1), Individual(c2)


def round_block_crossover(p1, p2, tournament):
    NR = tournament.num_rounds
    bs = random.randint(3, max(3, NR // 4))
    block = set(range(random.randint(1, NR - bs + 1), random.randint(1, NR - bs + 1) + bs))
    ts2 = _p2_ts(p2)
    c1, c2 = [], []
    for m in p1.matches:
        key = (m.home_team_id, m.away_team_id)
        t2 = ts2.get(key, m.timeslot_id)
        if m.round in block:
            c1.append(Match(m.home_team_id, m.away_team_id, m.round, t2))
            c2.append(Match(m.home_team_id, m.away_team_id, m.round, m.timeslot_id))
        else:
            c1.append(Match(m.home_team_id, m.away_team_id, m.round, m.timeslot_id))
            c2.append(Match(m.home_team_id, m.away_team_id, m.round, t2))
    return Individual(c1), Individual(c2)


def team_based_crossover(p1, p2, tournament):
    subset = set(random.sample(range(tournament.num_teams),
                               random.randint(2, max(2, tournament.num_teams // 3))))
    ts2 = _p2_ts(p2)
    c1, c2 = [], []
    for m in p1.matches:
        key = (m.home_team_id, m.away_team_id)
        t2 = ts2.get(key, m.timeslot_id)
        if m.home_team_id in subset or m.away_team_id in subset:
            c1.append(Match(m.home_team_id, m.away_team_id, m.round, t2))
            c2.append(Match(m.home_team_id, m.away_team_id, m.round, m.timeslot_id))
        else:
            c1.append(Match(m.home_team_id, m.away_team_id, m.round, m.timeslot_id))
            c2.append(Match(m.home_team_id, m.away_team_id, m.round, t2))
    return Individual(c1), Individual(c2)


def round_swap_crossover(p1, p2, tournament):
    NR = tournament.num_rounds
    r1, r2 = random.randint(1, NR), random.randint(1, NR)
    ts2 = _p2_ts(p2)
    mp2 = _p2_map(p2)
    c1, c2 = [], []
    for m in p1.matches:
        key = (m.home_team_id, m.away_team_id)
        t2 = ts2.get(key, m.timeslot_id)
        if m.round in (r1, r2):
            c1.append(Match(m.home_team_id, m.away_team_id, m.round, t2))
        p2m = mp2.get(key)
        if m.round in (r1, r2) and p2m:
            c1.append(Match(m.home_team_id, m.away_team_id, p2m.round, p2m.timeslot_id))
            c2.append(Match(m.home_team_id, m.away_team_id, m.round, m.timeslot_id))
        else:
            c1.append(Match(m.home_team_id, m.away_team_id, m.round, m.timeslot_id))
            c2.append(Match(m.home_team_id, m.away_team_id, m.round, t2))
            c2.append(Match(m.home_team_id, m.away_team_id, 
                            p2m.round if p2m else m.round, 
                            p2m.timeslot_id if p2m else m.timeslot_id))
    c1i, c2i = Individual(c1), Individual(c2)
    repair_individual(c1i, tournament); repair_individual(c2i, tournament)
    return c1i, c2i
    return Individual(c1), Individual(c2)


def structural_round_crossover(p1, p2, tournament):
    NR = tournament.num_rounds
    bs = random.randint(2, max(2, NR // 5))
    block = set(range(random.randint(1, NR - bs + 1), random.randint(1, NR - bs + 1) + bs))
    mp1, mp2 = _p2_map(p1), _p2_map(p2)
    c1 = [copy.deepcopy(m) for m in p1.matches]
    c2 = [copy.deepcopy(m) for m in p2.matches]
    for m in c1:
        key = (m.home_team_id, m.away_team_id)
        if m.round in block and key in mp2:
            m.round, m.timeslot_id = mp2[key].round, mp2[key].timeslot_id
    for m in c2:
        key = (m.home_team_id, m.away_team_id)
        if m.round in block and key in mp1:
            m.round, m.timeslot_id = mp1[key].round, mp1[key].timeslot_id
    return Individual(c1), Individual(c2)


def round_match_swap_crossover(p1, p2, tournament):
    NR = tournament.num_rounds
    rnd = random.randint(1, NR)
    mp2 = _p2_map(p2)
    c1, c2 = [], []
    for m in p1.matches:
        key = (m.home_team_id, m.away_team_id)
        p2m = mp2.get(key)
        if m.round == rnd and p2m:
            c1.append(Match(m.home_team_id, m.away_team_id, p2m.round, p2m.timeslot_id))
            c2.append(Match(m.home_team_id, m.away_team_id, m.round, m.timeslot_id))
        else:
            c1.append(copy.deepcopy(m))
            c2.append(Match(m.home_team_id, m.away_team_id, m.round,
                            p2m.timeslot_id if p2m else m.timeslot_id))
    c1i, c2i = Individual(c1), Individual(c2)
    repair_individual(c1i, tournament); repair_individual(c2i, tournament)
    return c1i, c2i


def single_match_crossover(p1, p2, tournament):
    mp2 = _p2_map(p2)
    if not p1.matches:
        return Individual([]), Individual([])
    sel = random.choice(p1.matches)
    key0 = (sel.home_team_id, sel.away_team_id)
    p2m0 = mp2.get(key0)
    c1, c2 = [], []
    for m in p1.matches:
        key = (m.home_team_id, m.away_team_id)
        p2m = mp2.get(key)
        if key == key0 and p2m0:
            c1.append(Match(m.home_team_id, m.away_team_id, p2m0.round, p2m0.timeslot_id))
            c2.append(Match(m.home_team_id, m.away_team_id, m.round, m.timeslot_id))
        else:
            c1.append(copy.deepcopy(m))
            c2.append(Match(m.home_team_id, m.away_team_id, m.round,
                            p2m.timeslot_id if p2m else m.timeslot_id))
    c1i, c2i = Individual(c1), Individual(c2)
    repair_individual(c1i, tournament); repair_individual(c2i, tournament)
    return c1i, c2i


def random_match_set_crossover(p1, p2, tournament):
    mp2 = _p2_map(p2)
    size = len(p1.matches)
    n_swap = random.randint(max(1, size // 10), max(3, size // 4))
    swap_keys = {(m.home_team_id, m.away_team_id)
                 for m in random.choices(p1.matches, k=n_swap)}
    c1, c2 = [], []
    for m in p1.matches:
        key = (m.home_team_id, m.away_team_id)
        p2m = mp2.get(key)
        if key in swap_keys and p2m:
            c1.append(Match(m.home_team_id, m.away_team_id, p2m.round, p2m.timeslot_id))
            c2.append(Match(m.home_team_id, m.away_team_id, m.round, m.timeslot_id))
        else:
            c1.append(copy.deepcopy(m))
            c2.append(Match(m.home_team_id, m.away_team_id, m.round,
                            p2m.timeslot_id if p2m else m.timeslot_id))
    c1i, c2i = Individual(c1), Individual(c2)
    repair_individual(c1i, tournament); repair_individual(c2i, tournament)
    return c1i, c2i


def adjusted_round_swapping_crossover(p1, p2, tournament):
    NR = tournament.num_rounds
    r1 = random.randint(1, NR)
    r2 = r1 % NR + 1 if (r2 := random.randint(1, NR)) == r1 else r2
    mp2 = _p2_map(p2)
    c1, c2 = [], []
    for m in p1.matches:
        key = (m.home_team_id, m.away_team_id)
        p2m = mp2.get(key)
        if m.round == r1 and p2m:
            c1.append(Match(m.home_team_id, m.away_team_id, p2m.round, p2m.timeslot_id))
            c2.append(Match(m.home_team_id, m.away_team_id, m.round, m.timeslot_id))
        else:
            c1.append(copy.deepcopy(m))
            c2.append(Match(m.home_team_id, m.away_team_id, m.round,
                            p2m.timeslot_id if p2m else m.timeslot_id))
    c1i, c2i = Individual(c1), Individual(c2)
    repair_individual(c1i, tournament); repair_individual(c2i, tournament)
    return c1i, c2i


def crossover(p1, p2, tournament):
    choice = random.choices(list(CROSSOVER_WEIGHTS), weights=list(CROSSOVER_WEIGHTS.values()), k=1)[0]
    dispatch = {
        "timeslot":            lambda: timeslot_crossover(p1, p2),
        "round_block":         lambda: round_block_crossover(p1, p2, tournament),
        "team_based":          lambda: team_based_crossover(p1, p2, tournament),
        "round_swap":          lambda: round_swap_crossover(p1, p2, tournament),
        "match_swap":          lambda: match_swap_crossover(p1, p2, tournament),
        "round_block_full_swap": lambda: round_block_full_swap_crossover(p1, p2, tournament),
        "round_match_swap":    lambda: round_match_swap_crossover(p1, p2, tournament),
        "single_match":        lambda: single_match_crossover(p1, p2, tournament),
        "random_match_set":    lambda: random_match_set_crossover(p1, p2, tournament),
        "adjusted_round_swap": lambda: adjusted_round_swapping_crossover(p1, p2, tournament),
        "structural_round":    lambda: structural_round_crossover(p1, p2, tournament),
    }
    return dispatch[choice]()


# ============================================================
# MUTATION
# ============================================================

def match_exchange_mutation(individual, tournament):
    matches = individual.matches
    if len(matches) < 2:
        return individual
    i, j = random.sample(range(len(matches)), 2)
    mi, mj = matches[i], matches[j]
    mi.round, mj.round = mj.round, mi.round
    mi.timeslot_id, mj.timeslot_id = mj.timeslot_id, mi.timeslot_id
    repair_individual(individual, tournament)
    return individual


def leg_swap_mutation(individual, tournament):
    matches = individual.matches
    first, second = {}, {}
    for i, m in enumerate(matches):
        key = (min(m.home_team_id, m.away_team_id), max(m.home_team_id, m.away_team_id))
        (first if m.home_team_id < m.away_team_id else second)[key] = i
    common = [k for k in first if k in second]
    if not common:
        return individual
    key = random.choice(common)
    m1, m2 = matches[first[key]], matches[second[key]]
    m1.round, m2.round = m2.round, m1.round
    m1.timeslot_id, m2.timeslot_id = m2.timeslot_id, m1.timeslot_id
    return individual


def swap_two_rounds_mutation(individual, tournament):
    rounds = sorted(set(m.round for m in individual.matches))
    if len(rounds) < 2:
        return individual
    r1, r2 = random.sample(rounds, 2)
    for m in individual.matches:
        if m.round == r1:   m.round = r2
        elif m.round == r2: m.round = r1
    return individual


def swap_rounds_between_teams(individual, tournament):
    matches = individual.matches
    NR = tournament.num_rounds
    a, b = random.sample(range(tournament.num_teams), 2)
    r1, r2 = random.sample(range(1, NR + 1), 2)
    for m in matches:
        if m.home_team_id in (a, b) or m.away_team_id in (a, b):
            if m.round == r1:   m.round = r2
            elif m.round == r2: m.round = r1
    return individual


def timeslot_mutation(individual, num_timeslots):
    idx = random.randint(0, len(individual.matches) - 1)
    individual.matches[idx].timeslot_id = random.randint(0, num_timeslots - 1)
    return individual


def block_timeslot_mutation(individual, tournament):
    num_ts = len(tournament.timeslots)
    rnd = random.choice(list(set(m.round for m in individual.matches)))
    new_ts = random.randint(0, num_ts - 1)
    for m in individual.matches:
        if m.round == rnd:
            m.timeslot_id = new_ts
    return individual


def round_reassign_mutation(individual, tournament):
    matches = individual.matches
    NR = tournament.num_rounds
    trm = _build_team_round_map(matches)
    for _ in range(20):
        idx = random.randint(0, len(matches) - 1)
        m = matches[idx]
        busy_h = set(trm.get(m.home_team_id, {}).keys()) - {m.round}
        busy_a = set(trm.get(m.away_team_id, {}).keys()) - {m.round}
        free = [r for r in range(1, NR + 1) if r not in busy_h and r not in busy_a and r != m.round]
        if free:
            matches[idx].round = random.choice(free)
            return individual
    return individual


def home_away_swap_mutation(individual):
    m = individual.matches[random.randint(0, len(individual.matches) - 1)]
    m.home_team_id, m.away_team_id = m.away_team_id, m.home_team_id
    return individual


def multi_point_mutation(individual, tournament):
    num_ts = len(tournament.timeslots)
    for idx in random.sample(range(len(individual.matches)),
                             random.randint(2, min(5, len(individual.matches)))):
        individual.matches[idx].timeslot_id = random.randint(0, num_ts - 1)
    return individual


def targeted_fix_mutation(individual, tournament):
    soft = count_soft_penalties(individual, tournament)
    if not soft:
        return individual
    worst = max(soft, key=lambda k: soft[k])

    if worst == "no_consec_big_team":
        for _ in range(3):
            individual = break_big_match_chain_mutation(individual, tournament)
        return individual

    matches = individual.matches
    num_ts = len(tournament.timeslots)
    prime_slots = get_prime_timeslots(tournament)

    if worst == "no_timeslot_overload":
        count = defaultdict(list)
        for i, m in enumerate(matches):
            count[(m.round, m.timeslot_id)].append(i)
        for (_, ts), idxs in count.items():
            if len(idxs) > 3:
                for i in idxs[3:]:
                    available = [t for t in range(num_ts) if t != ts]
                    matches[i].timeslot_id = random.choice(available)
                break

    elif worst == "prime_timeslot_fairness":
        prime_set = set(prime_slots)
        mid = tournament.num_rounds // 2
        prime_count = defaultdict(lambda: [0, 0])
        for m in matches:
            if m.timeslot_id in prime_set:
                half = 0 if m.round <= mid else 1
                prime_count[m.home_team_id][half] += 1
                prime_count[m.away_team_id][half] += 1
        for team_id in range(tournament.num_teams):
            for half in range(2):
                if prime_count[team_id][half] < 2:
                    cands = [i for i, m in enumerate(matches)
                             if (m.home_team_id == team_id or m.away_team_id == team_id)
                             and (m.round <= mid if half == 0 else m.round > mid)
                             and m.timeslot_id not in prime_set]
                    if cands:
                        matches[random.choice(cands)].timeslot_id = random.choice(prime_slots)
                    break
    else:
        for idx in random.sample(range(len(matches)), min(random.randint(1, 4), len(matches))):
            matches[idx].timeslot_id = random.randint(0, num_ts - 1)

    return individual


def break_big_match_chain_mutation(individual, tournament):
    matches = individual.matches
    NR = tournament.num_rounds
    trm = _build_team_round_map(matches)
    team_big = defaultdict(list)
    for i, m in enumerate(matches):
        if is_big_match(tournament, m.home_team_id, m.away_team_id):
            for t in (m.home_team_id, m.away_team_id):
                team_big[t].append((m.round, i))

    cands = [t for t, lst in team_big.items() if len(lst) >= 2]
    if not cands:
        return individual

    for _ in range(5):
        lst = sorted(team_big[random.choice(cands)])
        for k in range(len(lst) - 1):
            r1, i1 = lst[k]
            r2, i2 = lst[k + 1]
            if r2 == r1 + 1:
                m = matches[random.choice([i1, i2])]
                busy_h = set(trm.get(m.home_team_id, {}).keys()) - {m.round}
                busy_a = set(trm.get(m.away_team_id, {}).keys()) - {m.round}
                free = [r for r in range(1, NR + 1)
                        if r not in busy_h and r not in busy_a and r != m.round]
                if free:
                    m.round = random.choice(free)
                    # Ưu tiên chọn vòng trống gần nhất như mô tả trong báo cáo
                    m.round = min(free, key=lambda r: abs(r - m.round))
                    return individual
    return individual


def home_swap_same_round_mutation(individual, tournament):
    matches = individual.matches
    groups = defaultdict(list)
    for idx, m in enumerate(matches):
        groups[m.round].append(idx)
    if not groups:
        return individual
    group = groups[random.choice(list(groups))]
    if len(group) < 2:
        return individual
    i, j = random.sample(group, 2)
    matches[i].home_team_id, matches[j].home_team_id = matches[j].home_team_id, matches[i].home_team_id
    return individual


def pair_swap_mutation(individual, tournament):
    matches = individual.matches
    if len(matches) < 2:
        return individual
    i, j = random.sample(range(len(matches)), 2)
    mi, mj = matches[i], matches[j]
    mi.home_team_id, mj.home_team_id = mj.home_team_id, mi.home_team_id
    mi.away_team_id, mj.away_team_id = mj.away_team_id, mi.away_team_id
    return individual


def cyclic_shift_mutation(individual, tournament):
    matches = individual.matches
    if len(matches) < 3:
        return individual
    i1, i2, i3 = random.sample(range(len(matches)), 3)
    m1, m2, m3 = matches[i1], matches[i2], matches[i3]
    m1.home_team_id, m2.home_team_id, m3.home_team_id = m2.home_team_id, m3.home_team_id, m1.home_team_id
    m1.away_team_id, m2.away_team_id, m3.away_team_id = m2.away_team_id, m3.away_team_id, m1.away_team_id
    return individual


def shuffle_round_mutation(individual, tournament):
    matches = individual.matches
    groups = defaultdict(list)
    for idx, m in enumerate(matches):
        groups[m.round].append(idx)
    if not groups:
        return individual
    idxs = groups[random.choice(list(groups))]
    if len(idxs) < 2:
        return individual
    teams = []
    for idx in idxs:
        teams += [matches[idx].home_team_id, matches[idx].away_team_id]
    random.shuffle(teams)
    half = len(teams) // 2
    for k, idx in enumerate(idxs):
        matches[idx].home_team_id = teams[k]
        matches[idx].away_team_id = teams[half + k]
    return individual


def swap_team_identities_mutation(individual, tournament):
    a, b = random.sample(range(tournament.num_teams), 2)
    for m in individual.matches:
        if m.home_team_id == a:        m.home_team_id = b
        elif m.home_team_id == b:      m.home_team_id = a
        if m.away_team_id == a:        m.away_team_id = b
        elif m.away_team_id == b:      m.away_team_id = a
    return individual


def block_reverse_mutation(individual, tournament):
    NR = tournament.num_rounds
    r1 = random.randint(1, NR)
    r2 = random.randint(r1, NR)
    if r1 == r2:
        return individual
    for m in individual.matches:
        if r1 <= m.round <= r2:
            m.round = r1 + r2 - m.round
    return individual


def rotate_rounds_mutation(individual, tournament):
    NR = tournament.num_rounds
    shift = random.randint(1, NR - 1)
    for m in individual.matches:
        m.round = ((m.round - 1 + shift) % NR) + 1
    return individual


def mutate(individual, tournament, mutation_rate, aggressive=False):
    if random.random() >= mutation_rate:
        return individual
    individual = copy.deepcopy(individual)
    num_ts = len(tournament.timeslots)
    weights = MUTATION_WEIGHTS_AGGRESSIVE if aggressive else MUTATION_WEIGHTS_NORMAL

    dispatch = {
        "timeslot":             lambda: timeslot_mutation(individual, num_ts),
        "block_timeslot":       lambda: block_timeslot_mutation(individual, tournament),
        "round_reassign":       lambda: round_reassign_mutation(individual, tournament),
        "home_away_swap":       lambda: home_away_swap_mutation(individual),
        "multi_point":          lambda: multi_point_mutation(individual, tournament),
        "swap_rounds":          lambda: swap_rounds_between_teams(individual, tournament),
        "swap_two_rounds":      lambda: swap_two_rounds_mutation(individual, tournament),
        "targeted_fix":         lambda: targeted_fix_mutation(individual, tournament),
        "break_big_chain":      lambda: break_big_match_chain_mutation(individual, tournament),
        "match_exchange":       lambda: match_exchange_mutation(individual, tournament),
        "leg_swap":             lambda: leg_swap_mutation(individual, tournament),
        "home_swap_same_round": lambda: home_swap_same_round_mutation(individual, tournament),
        "pair_swap":            lambda: pair_swap_mutation(individual, tournament),
        "cyclic_shift":         lambda: cyclic_shift_mutation(individual, tournament),
        "shuffle_round":        lambda: shuffle_round_mutation(individual, tournament),
        "swap_team_identities": lambda: swap_team_identities_mutation(individual, tournament),
        "block_reverse":        lambda: block_reverse_mutation(individual, tournament),
        "rotate_rounds":        lambda: rotate_rounds_mutation(individual, tournament),
    }

    choice = random.choices(list(weights), weights=list(weights.values()), k=1)[0]
    individual = dispatch.get(choice, lambda: individual)()

    # 25% cơ hội áp thêm một mutation phụ
    if random.random() < 0.25:
        extra = random.choices(list(weights), weights=list(weights.values()), k=1)[0]
        if extra in ("shuffle_round", "swap_team_identities", "cyclic_shift"):
            individual = dispatch[extra]()

    return individual


# ============================================================
# DIVERSITY HELPERS
# ============================================================

def _individual_signature(individual):
    return tuple(sorted(
        (m.home_team_id, m.away_team_id, m.round, m.timeslot_id)
        for m in individual.matches
    ))


def _population_diversity(population):
    return len(set(_individual_signature(ind) for ind in population)) / len(population)


def _timeslot_diversity(population):
    sigs = set()
    for ind in population:
        sigs.add(tuple(m.timeslot_id for m in
                       sorted(ind.matches, key=lambda m: (m.home_team_id, m.away_team_id))))
    return len(sigs) / len(population)


def partial_restart(population, tournament, restart_count, elitism_count):
    sorted_pop = sorted(population, key=lambda ind: ind.fitness or -9999999, reverse=True)
    survivors = sorted_pop[:elitism_count]
    kept = sorted_pop[elitism_count: len(population) - restart_count]
    new_inds = []
    for _ in range(restart_count):
        ind = _create_round_robin_individual(tournament)
        repair_individual(ind, tournament)
        local_search_big_match_chain(ind, tournament, max_iter=15)
        repair_individual(ind, tournament)
        calculate_fitness(ind, tournament)
        new_inds.append(ind)
    return survivors + kept + new_inds


# ============================================================
# VÒNG LẶP CHÍNH
# ============================================================

def run_ga(tournament, params=None):
    start_time = time.time()
    p = params or DEFAULT_PARAMS
    pop_size      = p["population_size"]
    num_gen       = p["num_generations"]
    cx_rate       = p["crossover_rate"]
    mut_rate      = p["mutation_rate"]
    tourn_size    = p["tournament_size"]
    elitism       = p["elitism_count"]
    stag_lim      = p["stagnation_limit"]
    restart_lim   = p["restart_limit"]
    ls_interval   = p.get("local_search_interval", 4)
    ls_topk       = p.get("local_search_top_k", 12)

    population = create_population(tournament, pop_size)
    print("Repair + tính fitness quần thể ban đầu...")
    for ind in population:
        repair_individual(ind, tournament)
        calculate_fitness(ind, tournament)

    soft_init = count_soft_penalties(population[0], tournament)
    print(f"Soft penalty mẫu — no_consec_big_team: {soft_init.get('no_consec_big_team', 0)}")

    best = max(population, key=lambda ind: ind.fitness)
    history, stag = [], 0
    print(f"Fitness ban đầu tốt nhất: {best.fitness}")
    print(f"Bắt đầu tiến hóa ({num_gen} thế hệ)...\n")

    for gen in range(1, num_gen + 1):
        aggressive = stag >= stag_lim
        cur_mut = min(0.5, mut_rate * (1 + stag / stag_lim)) if aggressive else mut_rate

        if gen % ls_interval == 0:
            for elite in sorted(population, key=lambda ind: ind.fitness, reverse=True)[:ls_topk]:
                local_search_timeslot(elite, tournament, max_iter=25)
                local_search_round_swap(elite, tournament, max_iter=15)
                local_search_big_match_chain(elite, tournament, max_iter=20)
                repair_individual(elite, tournament)
                calculate_fitness(elite, tournament)

        if stag > 0 and stag % restart_lim == 0:
            population = partial_restart(population, tournament, pop_size - elitism, elitism)
            print(f"  [Gen {gen}] FULL RESTART (stagnation={stag})")

        sorted_pop = sorted(population, key=lambda ind: ind.fitness, reverse=True)
        new_pop = [copy.deepcopy(ind) for ind in sorted_pop[:elitism]]

        while len(new_pop) < pop_size:
            if random.random() < 0.7:
                p1 = tournament_selection(population, tourn_size)
                p2 = tournament_selection(population, tourn_size)
            else:
                p1 = rank_based_selection(population)
                p2 = rank_based_selection(population)

            c1, c2 = crossover(p1, p2, tournament) if random.random() < cx_rate \
                else (copy.deepcopy(p1), copy.deepcopy(p2))

            c1 = mutate(c1, tournament, cur_mut, aggressive)
            c2 = mutate(c2, tournament, cur_mut, aggressive)
            for c in (c1, c2):
                repair_individual(c, tournament)
                calculate_fitness(c, tournament)
            new_pop.extend([c1, c2])

        population = new_pop[:pop_size]

        for ind in population[:elitism * 2]:
            repair_no_consec_big_team(ind, tournament)
            repair_big_match_alternation(ind, tournament)
            repair_individual(ind, tournament)
            calculate_fitness(ind, tournament)

        cur_best = max(population, key=lambda ind: ind.fitness)
        if cur_best.fitness > best.fitness:
            best, stag = copy.deepcopy(cur_best), 0
        else:
            stag += 1

        history.append(best.fitness)

        if gen % 2 == 0:
            avg = sum(ind.fitness for ind in population) / pop_size
            div = _population_diversity(population)
            ts_div = _timeslot_diversity(population)
            mode = " [AGGRESSIVE]" if aggressive else ""
            print(f"Gen {gen:4d} | Best: {best.fitness:8.0f} | Avg: {avg:8.0f} "
                  f"| Div: {div:.2f} TsDiv: {ts_div:.2f} | Mut: {cur_mut:.3f}{mode}")

        if best.fitness == 0:
            print(f"\nLịch hoàn hảo tại thế hệ {gen}!")
            break

    duration = time.time() - start_time
    print(f"\nKết thúc! Fitness tốt nhất: {best.fitness}")
    print(f"Thời gian thực hiện: {duration:.2f} giây")
    
    repair_individual(best, tournament)
    calculate_fitness(best, tournament)
    return best, history


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    import os, sys, json
    from datetime import datetime
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from models import Tournament

    base_dir = os.path.dirname(os.path.abspath(__file__))
    tournament = Tournament(os.path.join(base_dir, "../data/teams8.json"))

    best, history = run_ga(tournament, params={
        "population_size":       250,
        "num_generations":       300,
        "crossover_rate":        0.85,
        "mutation_rate":         0.15,
        "tournament_size":       7,
        "elitism_count":         6,
        "stagnation_limit":      8,
        "restart_limit":         35,
        "local_search_interval": 3,
        "local_search_top_k":    10,
    })

    print(f"\nSố trận: {len(best.matches)} | Fitness: {best.fitness}")

    hard = count_hard_violations(best, tournament)
    soft = count_soft_penalties(best, tournament)

    print("\n--- RÀNG BUỘC CỨNG ---")
    for k, v in hard.items():
        print(f"  {'✓' if v == 0 else '✗'} {k:35s}: {v}")

    soft_groups = {
    "F1 — Công bằng đội": [
        "home_away_balance",
        "home_distribution",
        "min_rest_days",
        "season_edge_balance",
    ],
    "F2 — Chất lượng trận": [
        "derby_distribution",
        "no_derby_same_round",
        "no_consec_big_team",
        "big_match_half_balance",
        "big_match_monthly",
        "big_match_alternation",
    ],
    "F3 — Vận hành/Phát sóng": [
        "prime_timeslot_fairness",
        "no_timeslot_overload",
        "travel_distance_balance",
    ],
}

    print("\n--- RÀNG BUỘC MỀM ---")
    total = 0
    for gname, keys in soft_groups.items():
        print(f"  [{gname}]")
        for k in keys:
            v = soft.get(k, 0); total += v
            print(f"    {'✓' if v == 0 else '-'} {k:28s}: {v}")
    print(f"\n  Tổng soft penalty: {total}")

    if history:
        mid = len(history) // 2
        print(f"\n--- TIẾN HOÁ ---")
        print(f"  Gen   1: {history[0]:.0f}")
        print(f"  Gen {mid:3d}: {history[mid]:.0f}")
        print(f"  Gen {len(history):3d}: {history[-1]:.0f}")
        print(f"  Cải thiện: {history[-1] - history[0]:.0f}")

    # Lưu kết quả
    result_dir = os.path.join(base_dir, "result")
    os.makedirs(result_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(os.path.join(result_dir, f"best_schedule_{ts}.json"), "w", encoding="utf-8") as f:
        json.dump({
            "tournament": getattr(tournament, "name", "Unknown"),
            "num_teams": tournament.num_teams,
            "fitness": best.fitness,
            "num_matches": len(best.matches),
            "timestamp": ts,
            "matches": [{"round": m.round, "timeslot_id": m.timeslot_id,
                         "home_team_id": m.home_team_id, "away_team_id": m.away_team_id}
                        for m in best.matches],
        }, f, ensure_ascii=False, indent=2)

    with open(os.path.join(result_dir, f"fitness_history_{ts}.json"), "w", encoding="utf-8") as f:
        json.dump({"history": history, "best_fitness": best.fitness,
                   "total_generations": len(history)}, f, ensure_ascii=False, indent=2)

    with open(os.path.join(result_dir, f"summary_{ts}.txt"), "w", encoding="utf-8") as f:
        f.write(f"=== KẾT QUẢ GA SCHEDULING ===\n")
        f.write(f"Thời gian : {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"Tournament: {getattr(tournament, 'name', 'Unknown')}\n")
        f.write(f"Số đội    : {tournament.num_teams} | Số trận: {len(best.matches)}\n")
        f.write(f"Fitness   : {best.fitness}\n\n")
        f.write("--- RÀNG BUỘC CỨNG ---\n")
        for k, v in hard.items():
            f.write(f"  {'✓' if v == 0 else '✗'} {k:35s}: {v}\n")
        f.write("\n--- RÀNG BUỘC MỀM ---\n")
        t2 = 0
        for gname, keys in soft_groups.items():
            f.write(f"  [{gname}]\n")
            for k in keys:
                v = soft.get(k, 0); t2 += v
                f.write(f"    {'✓' if v == 0 else '-'} {k:28s}: {v}\n")
        f.write(f"\n  Tổng soft penalty: {t2}\n")
        if history:
            f.write(f"\n--- TIẾN HOÁ ---\n")
            f.write(f"  Gen   1: {history[0]:.0f}\n")
            f.write(f"  Gen {mid:3d}: {history[mid]:.0f}\n")
            f.write(f"  Gen {len(history):3d}: {history[-1]:.0f}\n")
            f.write(f"  Cải thiện: {history[-1] - history[0]:.0f}\n")

    print(f"\n✅ Đã lưu kết quả vào: {result_dir}")