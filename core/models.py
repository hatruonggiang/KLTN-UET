import json
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime, timedelta


# ============================================================
# HÀM TIỆN ÍCH
# ============================================================

def haversine(lat1, lng1, lat2, lng2):
    """Tính khoảng cách (km) giữa 2 tọa độ trên bề mặt Trái Đất."""
    R = 6371
    dlat = radians(lat2 - lat1)
    dlng = radians(lng2 - lng1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlng/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))


# ============================================================
# CLASS TEAM
# ============================================================

class Team:
    def __init__(self, id, name, short_name, stadium, city, lat, lng, champ_potential=5):
        self.id              = id
        self.name            = name
        self.short_name      = short_name
        self.stadium         = stadium
        self.city            = city
        self.lat             = lat
        self.lng             = lng
        self.champ_potential = champ_potential

    def distance_to(self, other_team):
        """Tính khoảng cách (km) từ sân đội này đến sân đội khác."""
        return haversine(self.lat, self.lng, other_team.lat, other_team.lng)

    def __repr__(self):
        return f"Team({self.short_name} - {self.stadium}, potential={self.champ_potential})"


# ============================================================
# CLASS MATCH
# ============================================================

class Match:
    def __init__(self, home_team_id, away_team_id, round, timeslot_id):
        self.home_team_id = home_team_id
        self.away_team_id = away_team_id
        self.round        = round
        self.timeslot_id  = timeslot_id

    def __repr__(self):
        return (f"Match(home={self.home_team_id}, away={self.away_team_id}, "
                f"round={self.round}, timeslot={self.timeslot_id})")


# ============================================================
# CLASS INDIVIDUAL
# ============================================================

class Individual:
    def __init__(self, matches):
        self.matches           = matches
        self.fitness           = None
        self.objectives        = None
        self.is_feasible       = False
        self.rank              = 0
        self.crowding_distance = 0.0

    def __repr__(self):
        return f"Individual({len(self.matches)} matches, fitness={self.fitness})"


# ============================================================
# CLASS TOURNAMENT
# ============================================================

class Tournament:
    def __init__(self, filepath):
        """Load toàn bộ dữ liệu giải đấu từ file teams.json."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Load danh sách đội
        self.teams = [
            Team(
                id              = t["id"],
                name            = t["name"],
                short_name      = t["short_name"],
                stadium         = t["stadium"],
                city            = t["city"],
                lat             = t["lat"],
                lng             = t["lng"],
                champ_potential = t.get("champ_potential", 5),
            )
            for t in data["teams"]
        ]

        # Load timeslots — cấu trúc mới: {id, day_offset, hour}
        self.timeslots = data["timeslots"]          # list[dict]
        self.num_timeslots = len(self.timeslots)    # dùng trong ga.py khi random slot

        # Load thông tin giải đấu
        info = data["tournament"]
        self.name              = info["name"]
        self.num_teams         = info["num_teams"]
        self.num_rounds        = info["num_rounds"]
        self.matches_per_round = info["matches_per_round"]
        self.start_date        = datetime.strptime(info["start_date"], "%Y-%m-%d")
        self.end_date          = datetime.strptime(info["end_date"],   "%Y-%m-%d")
        self.days_per_round    = info.get(
            "days_per_round",
            (self.end_date - self.start_date).days // self.num_rounds
        )

    # ── Helpers cũ ───────────────────────────────────────────

    def get_team(self, team_id):
        """Lấy đội bóng theo ID."""
        return self.teams[team_id]

    def get_timeslot(self, timeslot_id):
        """Lấy thông tin timeslot theo ID."""
        return self.timeslots[timeslot_id]

    def get_distance(self, home_team_id, away_team_id):
        """Tính khoảng cách (km) giữa sân nhà và sân khách."""
        return self.get_team(home_team_id).distance_to(self.get_team(away_team_id))

    def get_champ_potential(self, team_id):
        """Lấy champ_potential của một đội theo ID."""
        return self.teams[team_id].champ_potential

    # ── Helper mới: datetime thật ────────────────────────────

    def get_match_datetime(self, round_num: int, timeslot_id: int) -> datetime:
        """
        Trả về datetime tuyệt đối của một trận.

        Công thức:
          round_start = start_date + (round_num - 1) * days_per_round
          match_dt    = round_start + day_offset ngày + hour giờ
        """
        ts          = self.timeslots[timeslot_id]
        round_start = self.start_date + timedelta(days=(round_num - 1) * self.days_per_round)
        return round_start + timedelta(days=ts["day_offset"], hours=ts["hour"])

    def __repr__(self):
        return (f"Tournament({self.name}, {self.num_teams} teams, "
                f"{self.num_rounds} rounds, "
                f"{self.start_date.date()} → {self.end_date.date()})")


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    import os
    base_dir   = os.path.dirname(os.path.abspath(__file__))
    tournament = Tournament(os.path.join(base_dir, "../data/teams.json"))

    print(tournament)
    print()

    for team in tournament.teams:
        print(f"{team.short_name:>4} | {team.name:<15} | Potential: {team.champ_potential}")

    print()
    dist = tournament.get_distance(0, 12)
    print(f"Khoảng cách Arsenal → Man United: {dist:.1f} km")
    print(f"Man City potential  : {tournament.get_champ_potential(11)}")
    print(f"Số timeslots        : {tournament.num_timeslots}")

    # Test get_match_datetime
    dt = tournament.get_match_datetime(round_num=1, timeslot_id=2)
    print(f"Vòng 1 - slot 2 (Sat 15h): {dt.strftime('%A %d/%m/%Y %Hh%M')}")

    dt2 = tournament.get_match_datetime(round_num=20, timeslot_id=4)
    print(f"Vòng 20 - slot 4 (Sat 20h): {dt2.strftime('%A %d/%m/%Y %Hh%M')}")

    match = Match(home_team_id=0, away_team_id=12, round=1, timeslot_id=2)
    print(match)