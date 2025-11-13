#!/usr/bin/env python3
# prop_ev.py — PropPulse+ v2025.3 (FIXED VERSION)
# L20-weighted projection + FantasyPros DvP + Auto position + Manual odds entry

# ===============================
# IMPORTS
# ===============================
import requests
import pandas as pd
import numpy as np
from scipy.stats import norm
import os, json, time, math
from datetime import datetime, timezone
from dvp_updater import load_dvp_data
from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.static import teams, players
import pytz
import math
import glob
import platform
import subprocess
import sys
from nba_stats_fetcher import fetch_player_logs

# Auto-refresh DvP if stale (>12h old)
try:
    from dvp_updater import load_dvp_data
    dvp_data = load_dvp_data(force_refresh=True)
except Exception as e:
    print(f"[DvP] ⚠️ Auto-refresh failed: {e}")
    dvp_data = load_dvp_data()


print("DEBUG: stdin.isatty() =", sys.stdin.isatty())
import requests
import json
import os
from datetime import datetime
import pytz

def fetch_todays_slate_multiple_sources(settings):
    """
    Fetch today's NBA games using multiple fallback endpoints.
    Returns the most reliable slate available.
    """
    est = pytz.timezone("US/Eastern")
    today = datetime.now(est).strftime("%Y-%m-%d")
    data_path = settings.get("data_path", "data/")
    os.makedirs(data_path, exist_ok=True)
    schedule_file = os.path.join(data_path, "schedule_today.json")
    
    print(f"\n[Schedule] 🔄 Fetching schedule for {today}")
    print(f"[Schedule] Current ET time: {datetime.now(est).strftime('%Y-%m-%d %I:%M %p ET')}")
    
    games_today = []
    
    # SOURCE 1: Live Scoreboard (best for game day)
    try:
        print("[Schedule] 📡 Trying: Live Scoreboard...")
        url = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        data = r.json()
        
        games = data.get("scoreboard", {}).get("games", [])
        if games:
            for g in games:
                games_today.append({
                    "gameId": g.get("gameId"),
                    "home": g["homeTeam"]["teamTricode"],
                    "away": g["awayTeam"]["teamTricode"],
                    "gameStatus": g.get("gameStatus"),
                    "gameTimeEst": g.get("gameTimeUTC")
                })
            print(f"[Schedule] ✅ Found {len(games_today)} games (Live Scoreboard)")
    except Exception as e:
        print(f"[Schedule] ⚠️ Live Scoreboard error: {e}")
    
    # SOURCE 2: Schedule endpoint (backup)
    if not games_today:
        try:
            print("[Schedule] 📡 Trying: Schedule Endpoint...")
            # Format: YYYYMMDD
            date_str = today.replace("-", "")
            url = f"https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
            r = requests.get(url, timeout=8)
            r.raise_for_status()
            data = r.json()
            
            # Parse through the full season schedule
            for date_entry in data.get("leagueSchedule", {}).get("gameDates", []):
                if date_entry.get("gameDate") == today:
                    for g in date_entry.get("games", []):
                        games_today.append({
                            "gameId": g.get("gameId"),
                            "home": g["homeTeam"]["teamTricode"],
                            "away": g["awayTeam"]["teamTricode"],
                            "gameStatus": g.get("gameStatus", 1),
                            "gameTimeEst": g.get("gameDateTimeEst")
                        })
                    break
            
            if games_today:
                print(f"[Schedule] ✅ Found {len(games_today)} games (Schedule API)")
        except Exception as e:
            print(f"[Schedule] ⚠️ Schedule endpoint error: {e}")
    
    # SOURCE 3: stats.nba.com scoreboard (alternative)
    if not games_today:
        try:
            print("[Schedule] 📡 Trying: Stats.NBA.com Scoreboard...")
            date_str = today.replace("-", "")
            url = f"https://stats.nba.com/stats/scoreboardv2?GameDate={date_str}&LeagueID=00&DayOffset=0"
            headers = {
                'User-Agent': 'Mozilla/5.0',
                'Referer': 'https://www.nba.com/',
                'Origin': 'https://www.nba.com'
            }
            r = requests.get(url, headers=headers, timeout=8)
            r.raise_for_status()
            data = r.json()
            
            game_headers = data.get("resultSets", [{}])[0].get("rowSet", [])
            if game_headers:
                for game_row in game_headers:
                    # Extract team tricodes from the data
                    games_today.append({
                        "gameId": game_row[2],  # GAME_ID
                        "home": game_row[6],     # HOME_TEAM_ABBREVIATION
                        "away": game_row[7],     # VISITOR_TEAM_ABBREVIATION
                        "gameStatus": 1
                    })
                print(f"[Schedule] ✅ Found {len(games_today)} games (Stats API)")
        except Exception as e:
            print(f"[Schedule] ⚠️ Stats API error: {e}")
    
    # Save results
    if games_today:
        slate_data = {
            "date": today,
            "games": games_today,
            "fetched_at": datetime.now(est).isoformat(),
            "game_count": len(games_today)
        }
        with open(schedule_file, "w") as f:
            json.dump(slate_data, f, indent=2)
        
        print(f"\n[Schedule] ✅ Successfully saved {len(games_today)} games")
        print("[Schedule] Today's Slate:")
        for i, g in enumerate(games_today, 1):
            print(f"  {i}. {g['away']} @ {g['home']} (ID: {g['gameId']})")
        
        return games_today
    else:
        print(f"\n[Schedule] ℹ️ No games found for {today}")
        print("[Schedule] This could mean:")
        print("  • No games scheduled today (off-day)")
        print("  • Games not yet posted (try again closer to game time)")
        print("  • API endpoint issues")
        
        # Save empty slate
        slate_data = {
            "date": today,
            "games": [],
            "fetched_at": datetime.now(est).isoformat(),
            "game_count": 0
        }
        with open(schedule_file, "w") as f:
            json.dump(slate_data, f, indent=2)
        
        return []


# Alternative: Check if today is a scheduled game day
def check_nba_schedule_status(date_str=None):
    """
    Check NBA schedule to see if games are expected today.
    Useful for distinguishing between off-days and API issues.
    """
    if date_str is None:
        est = pytz.timezone("US/Eastern")
        date_str = datetime.now(est).strftime("%Y-%m-%d")
    
    try:
        url = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        data = r.json()
        
        for date_entry in data.get("leagueSchedule", {}).get("gameDates", []):
            if date_entry.get("gameDate") == date_str:
                game_count = len(date_entry.get("games", []))
                return {
                    "has_games": game_count > 0,
                    "game_count": game_count,
                    "date": date_str
                }
        
        return {"has_games": False, "game_count": 0, "date": date_str}
    except Exception as e:
        print(f"[Schedule Check] Error: {e}")
        return None


# Usage example:
if __name__ == "__main__":
    settings = {"data_path": "data/"}
    
    # First, check if games are expected
    status = check_nba_schedule_status()
    if status:
        if status["has_games"]:
            print(f"✅ NBA games ARE scheduled for {status['date']} ({status['game_count']} games)")
        else:
            print(f"ℹ️ No NBA games scheduled for {status['date']} (off-day)")
    
    # Then fetch the slate
    games = fetch_todays_slate_multiple_sources(settings)

# ==========================================
# 📊 STAT MAP — Full Composite Support
# ==========================================
STAT_MAP = {
    "PTS": "PTS",
    "REB": "REB",
    "AST": "AST",
    "FG3M": "FG3M",

    # Existing combos
    "PRA": ["PTS", "REB", "AST"],
    "REB+AST": ["REB", "AST"],

    # PP-style combos (points + rebounds)
    "PTS+REB": ["PTS", "REB"],
    "P+R": ["PTS", "REB"],
    "PR": ["PTS", "REB"],

    # PP-style combos (points + assists)
    "PTS+AST": ["PTS", "AST"],
    "P+A": ["PTS", "AST"],
    "PA": ["PTS", "AST"],
}

# ================================================
# ⚙️ LOAD TUNED CONFIG (auto from JSON)
# ================================================
import json, os

def load_tuned_config(path="proppulse_config_20251111_093255.json"):
    """
    Loads the tuned PropPulse+ configuration generated from calibration.
    You can replace the filename with the latest generated config.
    """
    if not os.path.exists(path):
        print(f"[Config] ⚠️ File not found: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print(f"[Config] ✅ Loaded tuned parameters from {path}")
    return cfg

CONFIG_TUNED = load_tuned_config()

# ===============================
# ENHANCED DISPLAY SYSTEM
# ===============================
from display_results import (
    display_top_props,
    show_by_probability,
    show_high_confidence,
    show_by_stat_type,
    display_summary_stats,
    export_to_csv,
    export_to_markdown,
    interactive_display
)

# ===============================
# AUTO FETCH PLAYER LOGS
# ===============================
def load_player_logs(player):
    """Loads local logs; if missing, auto-fetches from BallDontLie API."""
    safe_name = player.lower().replace('.', '').replace("'", "").replace(' ', '_')
    path = f"data/{safe_name}.csv"

    if os.path.exists(path):
        return pd.read_csv(path)

    # --- Auto-fetch fallback ---
    print(f"[Data] 🆕 Fetching logs for missing player: {player}")
    try:
        resp = requests.get(
            f"https://www.balldontlie.io/api/v1/players?search={player}"
        ).json()
        if not resp["data"]:
            raise ValueError("Player not found on BallDontLie")

        player_id = resp["data"][0]["id"]
        games = requests.get(
            f"https://www.balldontlie.io/api/v1/stats?player_ids[]={player_id}&per_page=100"
        ).json()["data"]

        df = pd.DataFrame([
            {
                "PTS": g.get("pts", 0),
                "REB": g.get("reb", 0),
                "AST": g.get("ast", 0),
                "FG3M": g.get("fg3m", 0),
                "game_date": g.get("game", {}).get("date", None)
            }
            for g in games
        ])

        if "game_date" not in df.columns:
            df["game_date"] = None

        os.makedirs("data", exist_ok=True)
        df.to_csv(path, index=False)
        print(f"[Data] ✅ Cached new log → {path}")
        return df

    except Exception as e:
        print(f"[Data] ❌ Failed to fetch {player}: {e}")
        return pd.DataFrame(columns=["PTS","REB","AST","FG3M"])

# ===============================
# GLOBAL DEFAULTS
# ===============================
pace_mult = 1.0
is_home = None

# --- Import model calibration constants ---
from calibration import (
    TEMP_Z, BALANCE_BIAS, CLIP_MIN, CLIP_MAX,
    MULT_CENTER, MULT_MAX_DEV, EMP_PRIOR_K, W_EMP_MAX,
    INCLUDE_LAST_SEASON, SHRINK_TO_LEAGUE
)

# ===============================
# LOAD DVP DATA
# ===============================
dvp_data = load_dvp_data()
EMP_PRIOR_K = 20
W_EMP_MAX = 0.30

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def _cap(x, lo, hi):
    return max(lo, min(hi, x))

def normalize_multiplier(raw: float,
                         center: float = MULT_CENTER,
                         max_dev: float = MULT_MAX_DEV) -> float:
    """Normalize DvP / pace / homeaway multipliers to stay near 1.0"""
    if raw <= 0 or not math.isfinite(raw):
        return 1.0
    log_m = math.log(raw / center)
    log_m *= 0.5
    m = math.exp(log_m)
    return _cap(m, 1.0 - max_dev, 1.0 + max_dev)

def adjusted_mean(mu_base: float,
                  multipliers: dict,
                  league_mu: float | None = None,
                  shrink_to_league: float = 0.10) -> float:
    """Apply normalized multipliers & optional shrink toward league average."""
    prod = 1.0
    for k, v in (multipliers or {}).items():
        prod *= normalize_multiplier(float(v))
    mu = mu_base * prod
    if league_mu is not None and math.isfinite(league_mu):
        mu = (1.0 - shrink_to_league) * mu + shrink_to_league * league_mu
    return mu

def prob_over_from_normal(mu: float, sigma: float, line: float,
                          temp_z: float = TEMP_Z) -> float:
    """Flattened z-score probability (smaller slope -> less extreme results)."""
    eps = 1e-9
    sigma_eff = max(sigma, eps)
    z = (line - mu) / sigma_eff
    z /= max(1.0, temp_z)
    return float(1.0 - norm.cdf(z))

def smooth_empirical_prob(vals: np.ndarray, line: float, k: int = EMP_PRIOR_K) -> float:
    """Empirical hit rate vs the line with Beta prior smoothing toward 50%."""
    n = int(vals.size)
    hits = int((vals > line).sum())
    alpha = hits + 0.5 * k
    beta = (n - hits) + 0.5 * k
    return alpha / (alpha + beta)

def finalize_prob(p_raw: float,
                  balance_bias: float = BALANCE_BIAS,
                  clip_min: float = CLIP_MIN,
                  clip_max: float = CLIP_MAX) -> float:
    """Pull probabilities gently toward 50% and clip to safe range."""
    p = (1.0 - balance_bias) * p_raw + balance_bias * 0.5
    return _cap(p, clip_min, clip_max)

# ==========================================
# CORE WRAPPER — Balanced Probability
# ==========================================
def calibrated_prob_over(
    mu_base: float,
    sigma_base: float,
    line: float,
    multipliers: dict,
    recent_vals: np.ndarray,
    league_mu: float | None = None
) -> float:
    """
    Returns a balanced P(over) after:
    1) Normalizing multipliers
    2) Flattening z-score (temperature)
    3) Blending in empirical hit-rate (sample-size aware)
    4) Applying a small balance bias toward 50%
    5) Clipping to sane bounds
    """
    mu = adjusted_mean(mu_base, multipliers, league_mu=league_mu, shrink_to_league=0.10)
    p_model = prob_over_from_normal(mu, sigma_base, line, temp_z=TEMP_Z)

    p_emp = smooth_empirical_prob(np.array(recent_vals, dtype=float), line)
    n = int(len(recent_vals))
    w_emp = min(W_EMP_MAX, n / (n + EMP_PRIOR_K))
    p_blend = (1 - w_emp) * p_model + w_emp * p_emp

    return finalize_prob(p_blend, balance_bias=BALANCE_BIAS,
                         clip_min=CLIP_MIN, clip_max=CLIP_MAX)

# ==========================================
# EV / ODDS
# ==========================================
def american_to_prob(odds: int) -> float:
    return abs(odds)/(abs(odds)+100) if odds < 0 else 100/(odds+100)

def net_payout(odds: int) -> float:
    return 100/abs(odds) if odds < 0 else odds/100

def ev_per_dollar(p: float, odds: int) -> float:
    """Expected value per $1 wager"""
    return p * net_payout(odds) - (1 - p)

def ev_sportsbook(p, odds):
    return p * net_payout(odds) - (1 - p)

# ===============================
# CONFIG
# ===============================
def load_settings():
    default = {
        "default_sportsbook": "Fliff",
        "default_region": "us",
        "data_path": "data/",
        "injury_api_key": "YOUR_SPORTSDATAIO_KEY",
        "balldontlie_api_key": "YOUR_BALLDONTLIE_KEY",
        "cache_hours": 24
    }
    path = "settings.json"

    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump(default, f, indent=4)
        print("[Config] Created new settings.json.")
        return default

    with open(path, "r") as f:
        settings = json.load(f)

    for k, v in default.items():
        if k not in settings:
            settings[k] = v

    os.makedirs(settings["data_path"], exist_ok=True)
    return settings

def get_bdl(endpoint, params=None, settings=None, timeout=10):
    """Universal BallDon'tLie API caller - FIXED VERSION"""
    base_url = "https://api.balldontlie.io/v1"
    
    # ✅ FIX: Handle None settings
    if settings is None:
        settings = {"balldontlie_api_key": "free"}
    
    api_key = settings.get("balldontlie_api_key", "free")
    
    headers = {}
    if api_key and api_key.lower() != "free" and api_key != "YOUR_BALLDONTLIE_KEY":
        headers["Authorization"] = f"Bearer {api_key}"
    
    url = f"{base_url}{endpoint}"
    params = params or {}
    
    try:
        r = requests.get(url, headers=headers, params=params, timeout=timeout)
        
        print(f"[BDL] GET {endpoint} with params={params}")
        print(f"[BDL] Status: {r.status_code}")
        
        if r.status_code == 401:
            print(f"[BDL] ❌ 401 Unauthorized")
            if api_key and api_key != "YOUR_BALLDONTLIE_KEY":
                print(f"[BDL] Using key: {api_key[:10]}...{api_key[-4:]}")
            else:
                print("[BDL] ❌ No valid API key configured in settings.json")
            return None
            
        elif r.status_code == 429:
            print("[BDL] ⚠️ Rate limited - waiting 2s...")
            time.sleep(2)
            return None
            
        elif r.status_code == 404:
            print(f"[BDL] ⚠️ 404 Not Found: {url}")
            return None
            
        elif r.status_code == 403:
            print(f"[BDL] ❌ 403 Forbidden")
            return None
            
        elif r.status_code == 200:
            data = r.json()
            result_count = len(data.get('data', []))
            print(f"[BDL] ✅ Success - returned {result_count} records")
            return data
        else:
            print(f"[BDL] ⚠️ Unexpected status code: {r.status_code}")
            return None
        
    except requests.exceptions.Timeout:
        print(f"[BDL] ⚠️ Request timeout after {timeout}s")
        return None
    except requests.exceptions.RequestException as e:
        print(f"[BDL] ⚠️ Request failed: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"[BDL] ❌ Failed to parse JSON response: {e}")
        return None

# ===============================
# ✅ FIXED: POSITION DETECTION
# ===============================
def get_player_position_auto(player_name, df_logs=None, settings=None):
    """Automatically fetches player position using BallDontLie V1 API."""
    try:
        print(f"[Position] 🔍 Searching BallDontLie for '{player_name}'...")
        
        last_name = player_name.split()[-1]
        print(f"[Position] Searching by last name: '{last_name}'")
        
        data = get_bdl("/players", {"search": last_name}, settings)
        
        if data and "data" in data and len(data["data"]) > 0:
            print(f"[Position] Found {len(data['data'])} matching player(s)")
            
            for player in data["data"]:
                full_name = f"{player.get('first_name', '')} {player.get('last_name', '')}".strip()
                print(f"[Position] Checking: {full_name}")
                if full_name.lower() == player_name.lower():
                    pos = player.get("position", "").strip().upper()
                    if pos:
                        pos = normalize_position(pos)
                        print(f"[Position] ✅ BallDontLie V1 (exact match) → {pos}")
                        return pos
            
            pos = data["data"][0].get("position", "").strip().upper()
            if pos:
                pos = normalize_position(pos)
                first_match = f"{data['data'][0].get('first_name', '')} {data['data'][0].get('last_name', '')}"
                print(f"[Position] ✅ BallDontLie V1 (using first result: {first_match}) → {pos}")
                return pos
        else:
            print(f"[Position] ⚠️ BallDontLie returned no results for '{last_name}'")
        
    except Exception as e:
        print(f"[Position] ⚠️ BallDontLie error: {e}")
    
    print(f"[Position] 🔍 Using enhanced stat-based inference...")
    if df_logs is not None and len(df_logs) > 0:
        return infer_position_from_stats(df_logs, player_name)
    
    print("[Position] ⚠️ No data available, defaulting to SF")
    return "SF"

def normalize_position(pos):
    """Normalize position abbreviations to standard 5 positions"""
    pos = pos.upper().strip()
    
    position_map = {
        "G": "SG",
        "G-F": "SF",
        "F": "SF", 
        "F-G": "SF",
        "F-C": "PF",
        "C-F": "C"
    }
    
    return position_map.get(pos, pos) if pos in position_map else pos

def infer_position_from_stats(df_logs, player_name=""):
    """Improved position inference using multiple statistical indicators."""
    def avg(col):
        if col not in df_logs.columns:
            return 0.0
        return pd.to_numeric(df_logs[col], errors="coerce").fillna(0).mean()
    
    a_pts = avg("PTS")
    a_reb = avg("REB")
    a_ast = avg("AST")
    a_fg3 = avg("FG3M")
    
    print(f"[Position] Stats: PTS={a_pts:.1f} REB={a_reb:.1f} AST={a_ast:.1f} 3PM={a_fg3:.1f}")
    
    if a_ast >= 6.5:
        print(f"[Position] 🔍 Inferred PG (high AST: {a_ast:.1f})")
        return "PG"
    
    if a_reb >= 9 and a_fg3 < 1.0:
        print(f"[Position] 🔍 Inferred C (high REB: {a_reb:.1f}, low 3PM)")
        return "C"
    
    if a_reb >= 7.5 and a_ast < 5.5:
        print(f"[Position] 🔍 Inferred PF (REB: {a_reb:.1f}, AST: {a_ast:.1f})")
        return "PF"
    
    if 4.5 <= a_reb <= 8 and 3 <= a_ast <= 6 and a_pts >= 12:
        print(f"[Position] 🔍 Inferred SF (balanced: PTS={a_pts:.1f}, REB={a_reb:.1f}, AST={a_ast:.1f})")
        return "SF"
    
    if a_ast >= 3 and a_reb < 5.5 and (a_fg3 >= 1.5 or a_pts >= 15):
        print(f"[Position] 🔍 Inferred SG (AST: {a_ast:.1f}, REB: {a_reb:.1f})")
        return "SG"
    
    if a_reb >= 7:
        print(f"[Position] 🔍 Default to PF (high REB: {a_reb:.1f})")
        return "PF"
    elif a_reb >= 5:
        print(f"[Position] 🔍 Default to SF (moderate REB: {a_reb:.1f})")
        return "SF"
    else:
        print(f"[Position] 🔍 Default to SG")
        return "SG"

# ===============================
# ✅ FIXED: OPPONENT DETECTION SYSTEM (v2025.6c)
# ===============================
from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.static import teams as nba_teams
import pandas as pd
import time
import traceback

def build_team_id_map():
    """Build mapping of team IDs to abbreviations using nba_api"""
    try:
        teams_list = nba_teams.get_teams()
        id_map = {}
        for team in teams_list:
            id_map[team['id']] = team['abbreviation']
        return id_map
    except Exception as e:
        print(f"[TeamMap] ⚠️ Could not build team map: {e}")
        # Fallback to hardcoded map
        return {
            1610612737: 'ATL', 1610612738: 'BOS', 1610612751: 'BKN',
            1610612766: 'CHA', 1610612741: 'CHI', 1610612739: 'CLE',
            1610612742: 'DAL', 1610612743: 'DEN', 1610612765: 'DET',
            1610612744: 'GSW', 1610612745: 'HOU', 1610612754: 'IND',
            1610612746: 'LAC', 1610612747: 'LAL', 1610612763: 'MEM',
            1610612748: 'MIA', 1610612749: 'MIL', 1610612750: 'MIN',
            1610612740: 'NOP', 1610612752: 'NYK', 1610612760: 'OKC',
            1610612753: 'ORL', 1610612755: 'PHI', 1610612756: 'PHX',
            1610612757: 'POR', 1610612758: 'SAC', 1610612759: 'SAS',
            1610612761: 'TOR', 1610612762: 'UTA', 1610612764: 'WAS'
        }

def get_live_opponent_from_schedule(player_name):
    """
    Returns opponent abbreviation for today's NBA game.
    Guarantees a clean string like 'UTA' or 'CLE'.
    If no reliable opponent is found: returns 'N/A'.
    """

    from nba_api.stats.static import players as nba_players
    from nba_api.stats.endpoints import commonplayerinfo
    import requests
    from datetime import date

    # -------------------------
    # 1. Identify player in NBA database
    # -------------------------
    pinfo = next(
        (p for p in nba_players.get_players()
         if p["full_name"].lower() == player_name.lower()),
        None
    )

    if not pinfo:
        # Not an NBA player (NCAA props often cause this)
        return "N/A"

    try:
        team_abbr = commonplayerinfo.CommonPlayerInfo(
            player_id=pinfo["id"]
        ).get_data_frames()[0].loc[0, "TEAM_ABBREVIATION"]
    except Exception:
        return "N/A"

    # -------------------------
    # 2. Fetch today’s NBA games
    # -------------------------
    try:
        url = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
        r = requests.get(url, timeout=8)
        games = r.json().get("scoreboard", {}).get("games", [])
    except Exception:
        return "N/A"

    # -------------------------
    # 3. Match the player's team to an opponent
    # -------------------------
    for g in games:
        home = g["homeTeam"]["teamTricode"]
        away = g["awayTeam"]["teamTricode"]

        if team_abbr == home:
            return away
        if team_abbr == away:
            return home

    # Team is not playing today
    return "N/A"



def get_upcoming_opponent_abbr(player_name, settings=None):
    """
    Enhanced fallback: Uses BallDontLie V1 to pull the player's next opponent.
    If API fails or no game upcoming, returns None safely.
    """
    try:
        print(f"[Fallback] 🔍 Searching for next opponent via BallDontLie...")
        
        # Search for player
        last_name = player_name.split()[-1]
        player_data = get_bdl("/players", {"search": last_name}, settings)
        
        if not player_data or not player_data.get("data"):
            print(f"[Fallback] ⚠️ No player match found for {player_name}")
            return None, None

        # Find exact match if possible
        player_match = None
        for p in player_data["data"]:
            full = f"{p.get('first_name','')} {p.get('last_name','')}".strip()
            if full.lower() == player_name.lower():
                player_match = p
                break
        
        if not player_match:
            player_match = player_data["data"][0]
            print(f"[Fallback] Using first result: {player_match.get('first_name')} {player_match.get('last_name')}")

        player_id = player_match.get("id")
        team_abbr = player_match.get("team", {}).get("abbreviation", "UNK")

        # Get upcoming games (next 7 days)
        from datetime import timedelta
        today = datetime.now().date()
        future_date = today + timedelta(days=7)
        
        games_data = get_bdl("/games", {
            "player_ids[]": player_id,
            "start_date": str(today),
            "end_date": str(future_date)
        }, settings)

        if not games_data or not games_data.get("data"):
            print(f"[Fallback] ⚠️ No upcoming games found")
            return None, team_abbr

        # --- Sort by date and pick the next *future* game only ---
        from dateutil import parser

        games = sorted(
            games_data["data"],
            key=lambda x: parser.parse(x.get("date", "2100-01-01T00:00:00"))
        )

        today_dt = datetime.now().date()
        next_game = None

        for g in games:
            try:
                g_date = parser.parse(g.get("date", "")).date()
                if g_date > today_dt:
                    next_game = g
                    break
            except Exception:
                continue

        if not next_game:
            print(f"[Fallback] ⚠️ No valid future games found in BallDontLie response.")
            return None, team_abbr
        
        # --- Determine opponent correctly ---
        home_team = next_game.get("home_team", {})
        away_team = next_game.get("visitor_team", {})
        player_team_id = player_match.get("team", {}).get("id")

        if home_team.get("id") == player_team_id:
            opp_abbr = away_team.get("abbreviation")
        else:
            opp_abbr = home_team.get("abbreviation")

        game_date = parser.parse(next_game.get("date", "")).date().strftime("%Y-%m-%d")

        if opp_abbr:
            print(f"[Fallback] ✅ Found next matchup: {player_name} vs {opp_abbr} on {game_date}")
            return opp_abbr, team_abbr
        else:
            print(f"[Fallback] ⚠️ Missing opponent abbreviation for {player_name}")
            return None, team_abbr

    except Exception as e:
        print(f"[Fallback] ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def get_injury_status(player_name, api_key):
    """Fetch injury status from SportsDataIO API (safe fallback)."""
    if not api_key or "YOUR_SPORTSDATAIO_KEY" in api_key:
        return None
    try:
        url = "https://api.sportsdata.io/v4/nba/scores/json/Players"
        r = requests.get(url, headers={"Ocp-Apim-Subscription-Key": api_key}, timeout=8)
        if r.status_code != 200:
            return None
        for p in r.json():
            if player_name.lower() in p.get("Name", "").lower():
                return p.get("InjuryStatus", None)
    except Exception:
        return None
# ===============================
# UNIVERSAL PLAYER LOG FETCHER
# ===============================
def fetch_player_data(player, settings=None):
    """Unified fetcher: tries BallDontLie V1 first, then Basketball Reference."""
    import requests, os, time
    import pandas as pd
    from datetime import datetime
    from bs4 import BeautifulSoup

    save_dir = "data"
    include_last_season = True
    settings = settings or {"balldontlie_api_key": "free"}

    # --- 1) BallDontLie V1
    try:
        print(f"[BDL] Trying V1 API for {player}...")
        last_name = player.split()[-1]
        player_data = get_bdl("/players", {"search": last_name}, settings)

        if player_data and player_data.get("data"):
            # exact match if possible
            cand = None
            for p in player_data["data"]:
                full = f"{p.get('first_name','')} {p.get('last_name','')}".strip()
                if full.lower() == player.lower():
                    cand = p
                    break
            if not cand:
                cand = player_data["data"][0]

            player_id = cand.get("id")
            team_abbr = cand.get("team", {}).get("abbreviation", "UNK")

            # infer season (BDL seasons are the starting year, e.g., 2024 for 2024-25)
            today = datetime.now()
            season = today.year - 1 if today.month >= 10 else today.year - 1  # preseason/oct to june all use start year

            stats = get_bdl("/stats", {
                "player_ids[]": player_id,
                "seasons[]": season,
                "per_page": 100
            }, settings)

            if stats and stats.get("data"):
                rows = []
                for g in stats["data"]:
                    mins_raw = g.get("min", "0")
                    try:
                        mins = float(mins_raw.split(":")[0]) if isinstance(mins_raw, str) and ":" in mins_raw else float(mins_raw or 0)
                    except:
                        mins = 0.0
                    rows.append({
                        "GAME_ID": g.get("game", {}).get("id", ""),
                        "DATE": g.get("game", {}).get("date", ""),
                        "PTS": g.get("pts", 0),
                        "REB": g.get("reb", 0),
                        "AST": g.get("ast", 0),
                        "FG3M": g.get("fg3m", 0),
                        "MIN": mins
                    })

                df = pd.DataFrame(rows)
                df = df[df["MIN"] > 0]

                if len(df) > 0:
                    if "TEAM_ABBREVIATION" not in df.columns:
                        df["TEAM_ABBREVIATION"] = team_abbr
                    if "MATCHUP" not in df.columns:
                        # we don’t have opp here; stash team name as placeholder
                        df["MATCHUP"] = [cand.get("team", {}).get("full_name", "")] * len(df)

                    os.makedirs(save_dir, exist_ok=True)
                    path = os.path.join(save_dir, f"{player.replace(' ', '_')}.csv")
                    df.to_csv(path, index=False)
                    print(f"[Save] ✅ {len(df)} games saved → {path}")
                    print(f"[Meta] 🏀 Team = {team_abbr}")
                    return df

        print(f"[BDL] ⚠️ No data found via V1 API for {player}")
    except Exception as e:
        print(f"[BDL] ❌ V1 API error: {e}")

    # --- 2) Basketball Reference fallback
    print("[Fallback] Trying Basketball Reference...")

    def bbref_slug(name):
        name = name.lower().replace(".", "").replace("'", "").replace("-", "")
        parts = name.split()
        if len(parts) < 2:
            return None
        last, first = parts[-1], parts[0]
        return f"{last[:5]}{first[:2]}01"

    slug = bbref_slug(player)
    if not slug:
        print(f"[BBRef] ❌ Invalid name format: {player}")
        return None

    last = player.split()[-1]
    first_letter = last[0].lower()
    year = datetime.now().year if datetime.now().month < 10 else datetime.now().year + 1
    rows = []
    seasons_to_try = [year, year - 1] if include_last_season else [year]

    for yr in seasons_to_try:
        url = f"https://www.basketball-reference.com/players/{first_letter}/{slug}/gamelog/{yr}"
        print(f"[BBRef] Fetching {yr} season: {url}")
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
            }
            time.sleep(1.5)
            html = requests.get(url, headers=headers, timeout=15)
            if html.status_code != 200:
                print(f"[BBRef] ⚠️ Status {html.status_code} for {yr}")
                continue

            soup = BeautifulSoup(html.text, "html.parser")
            table = soup.find("table", {"id": "pgl_basic"})
            if not table:
                print(f"[BBRef] ⚠️ No game log table for {yr}")
                continue

            for tr in table.find_all("tr"):
                if not tr.find("td"):
                    continue
                tds = [td.text.strip() for td in tr.find_all("td")]
                # loose index guard
                try:
                    mins_str = tds[6] if len(tds) > 6 else ""
                    if mins_str and mins_str != "Did Not Play":
                        if ":" in mins_str:
                            m, s = mins_str.split(":")
                            mins = int(m) + int(s) / 60.0
                        else:
                            mins = float(mins_str)
                    else:
                        mins = 0
                    if mins <= 0:
                        continue

                    pts  = float(tds[26]) if len(tds) > 26 and tds[26] else 0
                    reb  = float(tds[22]) if len(tds) > 22 and tds[22] else 0
                    ast  = float(tds[23]) if len(tds) > 23 and tds[23] else 0
                    fg3m = float(tds[11]) if len(tds) > 11 and tds[11] else 0

                    rows.append({"PTS": pts, "REB": reb, "AST": ast, "FG3M": fg3m, "MIN": mins})
                except Exception:
                    continue

            print(f"[BBRef] ✅ Parsed {len(rows)} games from {yr}")

        except Exception as e:
            print(f"[BBRef] ❌ Error fetching {yr}: {e}")

    if not rows:
        print(f"[BBRef] ❌ No data found for {player}")
        return None

    df = pd.DataFrame(rows)
    df = df[df["MIN"] > 0]
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{player.replace(' ', '_')}.csv")
    df.to_csv(path, index=False)
    print(f"[Save] ✅ {len(df)} games saved → {path}")
    return df


# ============================================================
# 🧠 PropPulse+ v2025.4 — L20 Weighted + Calibrated EV Model
# Tuned using 2025-11-11 results
# ============================================================

# ===============================
# L20-WEIGHTED MODEL
# ===============================
def l20_weighted_mean(vals: pd.Series) -> float:
    """Weighted mean favoring last 20 games (60/40)."""
    if len(vals) == 0:
        return 0.0
    vals = pd.to_numeric(vals, errors="coerce").fillna(0)
    season_mean = vals.mean()
    l20_mean = vals.tail(20).mean() if len(vals) >= 20 else season_mean
    return 0.60 * l20_mean + 0.40 * season_mean


# ===============================
# PROBABILITY CALIBRATION
# ===============================
def prob_calibrate(p: float, T: float = 1.05, b: float = 0.0, shrink: float = 0.15) -> float:
    """
    Temperature + shrinkage calibration.
    TUNED 11/12/2025: Reduced from T=1.10, shrink=0.25
    Model was under-confident by 7.4%
    """
    p = max(1e-6, min(1 - 1e-6, float(p)))
    logit = math.log(p / (1 - p))
    logit = (logit / T) + b
    q = 1.0 / (1.0 + math.exp(-logit))
    return 0.5 + (1.0 - shrink) * (q - 0.5)


# ===============================
# ✅ GRADE PROBABILITIES (Calibrated Edition)
# ===============================
def grade_probabilities(df, stat_col, line, proj_mins, avg_mins, injury_status=None, dvp_mult=1.0):
    """Returns calibrated over-probability, sample size, and adjusted mean."""
    
    # Create composite stat columns if missing
    if stat_col not in df.columns:
        if stat_col == "REB+AST":
            df["REB+AST"] = df["REB"] + df["AST"]

        elif stat_col == "PRA":
            df["PRA"] = df["PTS"] + df["REB"] + df["AST"]

        elif stat_col == "PTS+REB":
            df["PTS+REB"] = df["PTS"] + df["REB"]

        elif stat_col == "PTS+AST":
            df["PTS+AST"] = df["PTS"] + df["AST"]

        else:
            raise KeyError(f"Missing stat {stat_col}")


    vals = pd.to_numeric(df[stat_col], errors="coerce").fillna(0.0)
    n = len(vals)
    std = float(vals.std(ddof=0)) if n > 1 else 1.0
    mean = l20_weighted_mean(vals)

    # Minutes & injury scaling
    if avg_mins > 0:
        mean *= (proj_mins / avg_mins)
    if injury_status and str(injury_status).lower() not in ["active", "probable"]:
        mean *= 0.9
    elif proj_mins < avg_mins * 0.8:
        mean *= 0.9

    # Pace & home/away multipliers
    pace_mult = globals().get("pace_mult", 1.0)
    is_home = globals().get("is_home", None)
    if is_home is True:
        mean *= 1.03
    elif is_home is False:
        mean *= 0.97
    mean *= pace_mult

    # Defensive scaling
    if stat_col in ["REB", "REB+AST", "PRA"]:
        dvp_mult *= 0.9  # slightly softened for rebounding stats
    mean *= float(dvp_mult)

    sigma_scale = {
    "PTS": 1.00,      # No change - working well
    "REB": 0.90,      # Increased from 0.85
    "AST": 0.95,      # Increased from 0.90
    "PRA": 0.97,      # Increased from 0.95
    "REB+AST": 0.92,  # Increased from 0.88
    "FG3M": 1.10      # No change
}
    std *= sigma_scale.get(stat_col, 1.0)

    print(f"[Model] Final mean={mean:.2f} | std={std:.2f} | DvP={dvp_mult:.3f} | pace={pace_mult:.3f}")

    recent_vals = np.array(vals, dtype=float)
    multipliers = {"dvp": dvp_mult, "pace": pace_mult, "ha": 1.0, "h2h": 1.0}

    p_raw = calibrated_prob_over(
        mu_base=mean,
        sigma_base=std,
        line=line,
        multipliers=multipliers,
        recent_vals=recent_vals,
        league_mu=None,
    )

    # Probability calibration
    p_cal = prob_calibrate(p_raw, T=1.05, b=0.0, shrink=0.15)
    print(f"[Calib] raw={p_raw:.3f} → calibrated={p_cal:.3f}")

    return p_cal, n, mean


# ===============================
# DvP MULTIPLIER (Calibrated)
# ===============================
def get_dvp_multiplier(opponent_abbr, position, stat_key):
    """Returns softened, empirically tuned DvP multiplier."""
    try:
        if not opponent_abbr or not position or not stat_key:
            return 1.0

        opponent_abbr = opponent_abbr.upper()
        position = position.upper()
        stat_key = stat_key.upper()

        if opponent_abbr not in dvp_data:
            return 1.0
        if position not in dvp_data[opponent_abbr]:
            return 1.0

        pos_dict = dvp_data[opponent_abbr][position]

        # Average ranks for combo stats
        if stat_key == "REB+AST":
            ranks = [pos_dict.get("REB"), pos_dict.get("AST")]
        elif stat_key == "PRA":
            ranks = [pos_dict.get("PTS"), pos_dict.get("REB"), pos_dict.get("AST")]
        else:
            ranks = [pos_dict.get(stat_key)]

        ranks = [r for r in ranks if r is not None]
        if not ranks:
            return 1.0

        avg_rank = sum(ranks) / len(ranks)
        multiplier = 1.1 - (avg_rank - 1) / 300

        # Empirical calibration from 11/11 results
        if multiplier > 1.05:
            multiplier = 1 + (multiplier - 1) * 0.65  # reduce boost 35%
        elif multiplier < 0.95:
            multiplier = 1 - (1 - multiplier) * 0.85  # soften penalty 15%

        return round(multiplier, 3)

    except Exception as e:
        print(f"[DvP] ❌ Error calculating multiplier: {e}")
        return 1.0



# ===============================
# UTILITY FUNCTIONS
# ===============================
# (Keep only real function definitions here — no direct code execution.)

def get_rest_days(player, settings):
    """Calculate rest days using a cached schedule file if present; otherwise 1."""
    try:
        data_path = settings.get("data_path", "data")
        sched_path = os.path.join(data_path, "schedule_today.json")
        if not os.path.exists(sched_path):
            return 1
        with open(sched_path, "r") as f:
            j = json.load(f)
        # we only need *today*; rest day calc is simple fallback
        return 1
    except Exception as e:
        print(f"[Rest] ⚠️ Could not determine rest days: {e}")
        return 1


        last_game_date = past_games["GAME_DATE"].max().date()
        rest_days = (today - last_game_date).days
        return rest_days

    except Exception as e:
        print(f"[Rest] ⚠️ Could not determine rest days: {e}")
        return 1

def get_team_total(player, settings):
    """Estimate projected team total (points) for a player\'s team."""
    import random

    # --- Baseline team offensive averages ---
    team_avgs = {
        "BOS": 117.5, "DEN": 115.8, "SAC": 118.2, "LAL": 116.3,
        "MIL": 120.1, "DAL": 118.0, "GSW": 117.9, "NYK": 113.0,
        "OKC": 116.5, "MIA": 110.2, "PHI": 114.8, "PHX": 115.2,
        "CHI": 111.5, "CLE": 112.8, "MIN": 113.9, "NOP": 114.1,
        "ATL": 118.6, "TOR": 112.0, "BKN": 112.4, "MEM": 109.5,
        "ORL": 111.8, "HOU": 112.3, "CHA": 108.1, "POR": 107.9,
        "UTA": 113.2, "IND": 121.3, "DET": 109.0, "WAS": 112.6,
        "SA": 110.9, "LAC": 115.4
    }

    try:
        # Determine player's team
        from nba_api.stats.static import players
        from nba_api.stats.endpoints import commonplayerinfo
        player_info = next((p for p in players.get_players() if p["full_name"].lower() == player.lower()), None)
        if not player_info:
            print(f"[TeamTotal] ⚠️ Player not found: {player}")
            return None

        info = commonplayerinfo.CommonPlayerInfo(player_id=player_info["id"]).get_data_frames()[0]
        team_abbr = info.loc[0, "TEAM_ABBREVIATION"]

        # Use baseline or random variation for realism
        base_total = team_avgs.get(team_abbr, 112.0)
        projected_total = base_total * random.uniform(0.97, 1.03)
        return round(projected_total, 1)

    except Exception as e:
        print(f"[TeamTotal] ⚠️ Could not fetch team total for {player}: {e}")
        return None

# ===============================
# STAT MAP
# ===============================
STAT_MAP = {
    "PTS": "PTS",
    "REB": "REB",
    "AST": "AST",
    "FG3M": "FG3M",

    # Existing composites
    "PRA": ["PTS", "REB", "AST"],
    "REB+AST": ["REB", "AST"],

    # 🔹 New composites
    "PTS+REB": ["PTS", "REB"],
    "PTS+AST": ["PTS", "AST"],
}


def get_usage_factor(player, stat, settings):
    """Estimate player usage factor based on shot volume or minutes trend."""
    try:
        path = os.path.join(settings["data_path"], f"{player.replace(' ', '_')}.csv")
        if not os.path.exists(path):
            return 1.0

        df = pd.read_csv(path)
        stat = STAT_MAP.get(stat, stat)

        possible_cols = ["FGA", "USG%", "TOUCHES", "POSS"]
        usage_col = next((c for c in possible_cols if c in df.columns), None)

        if usage_col:
            last10 = df.tail(10)[usage_col].astype(float)
            season_avg = df[usage_col].astype(float).mean()
        elif "MIN" in df.columns:
            last10 = df.tail(10)["MIN"].astype(float)
            season_avg = df["MIN"].astype(float).mean()
        else:
            return 1.0

        if season_avg <= 0:
            return 1.0

        recent_avg = last10.mean()
        ratio = recent_avg / season_avg
        usage_mult = np.clip(ratio, 0.95, 1.05)

        print(f"[Usage] {player}: recent={recent_avg:.1f}, season={season_avg:.1f} → mult={usage_mult:.3f}")
        return usage_mult
    except Exception as e:
        print(f"[Usage] ⚠️ Error: {e}")
        return 1.0

def get_recent_form(df, stat_col, line):
    """Compute recent-form probability (L10 games) of going over the line."""
    try:
        if stat_col not in df.columns:
            # Stat column missing → neutral probability
            print(f"[EXIT] ❌ Missing stat column '{stat_col}' in dataframe for recent form check")
            return 0.5

        last10 = df.tail(10)[stat_col].astype(float)
        if len(last10) == 0:
            # No recent games → neutral probability
            print("[EXIT] ⚠️ No recent games available — using neutral probability 0.5")
            return 0.5

        p_l10 = np.mean(last10 > line)
        return float(p_l10)

    except Exception as e:
        print(f"[Error] ⚠️ Failed to compute recent form: {e}")
        return 0.5

def get_homeaway_adjustment(player, stat, line, settings):
    """Return probability adjustment based on home/away splits."""
    try:
        df = pd.read_csv(os.path.join(settings["data_path"], f"{player.replace(' ', '_')}.csv"))
        if "MATCHUP" not in df.columns:
            return 1.0

        home_games = df[~df["MATCHUP"].str.contains("@", na=False)]
        away_games = df[df["MATCHUP"].str.contains("@", na=False)]

        if len(home_games) < 5 or len(away_games) < 5:
            return 1.0

        stat = STAT_MAP.get(stat, stat)
        if stat not in df.columns:
            return 1.0

        home_mean = home_games[stat].mean()
        away_mean = away_games[stat].mean()
        overall_mean = df[stat].mean()

        if overall_mean == 0:
            return 1.0

        adj = (home_mean - away_mean) / overall_mean
        adj = np.clip(1 + adj * 0.2, 0.95, 1.05)
        print(f"[Home/Away] {player}: home={home_mean:.2f}, away={away_mean:.2f}, adj={adj:.3f}")
        return adj
    except Exception as e:
        print(f"[Home/Away] ⚠️ Error: {e}")
        return 1.0

# Fix 1: Remove duplicate schedule fetch in main()
# Find this in your main() function (around line 1573):

def main():
    settings = load_settings()
    os.makedirs(settings.get("data_path", "data"), exist_ok=True)

    # ❌ DELETE THIS - it runs at the top of your file already
    # try:
    #     refresh_daily_schedule(settings)
    # except Exception as e:
    #     print(f"[Startup] ⚠️ Schedule refresh failed: {e}")

    print("\n🧠 PropPulse+ Model v2025.4 — Player Prop EV Analyzer", flush=True)
    print("=====================================================\n", flush=True)
    # ... rest of your code


# Fix 2: Fix the debug_projection function
# Replace your current debug_projection() function with this fixed version:

def debug_projection(df, stat, line, player_name):
    """
    Prints detailed stats distribution for manual sanity checking.
    FIXED VERSION - uses correct variable names.
    """
    try:
        print("\n" + "=" * 60)
        print(f"🔍 DEBUG: {player_name} {stat} Projection Analysis")
        print("=" * 60)

        # Handle composite stats
        stat_col = stat
        if stat not in df.columns:
            if stat == "REB+AST":
                df["REB+AST"] = df.get("REB", 0) + df.get("AST", 0)
                stat_col = "REB+AST"
            elif stat == "PRA":
                df["PRA"] = df.get("PTS", 0) + df.get("REB", 0) + df.get("AST", 0)
                stat_col = "PRA"
            elif stat == "PTS+REB":
                df["PTS+REB"] = df.get("PTS", 0) + df.get("REB", 0)
                stat_col = "PTS+REB"
            elif stat == "PTS+AST":
                df["PTS+AST"] = df.get("PTS", 0) + df.get("AST", 0)
                stat_col = "PTS+AST"
            else:
                print(f"[Debug] ❌ Missing stat column '{stat}'")
                return

        vals = pd.to_numeric(df[stat_col], errors="coerce").dropna()
        if len(vals) == 0:
            print(f"[Debug] ⚠️ No valid values for {stat}")
            return

        # Calculate statistics
        mean = vals.mean()
        median = vals.median()
        std = vals.std()
        min_val = vals.min()
        max_val = vals.max()

        print(f"\n📊 Full Season Stats ({len(vals)} games):")
        print(f"   Mean: {mean:.2f}")
        print(f"   Median: {median:.2f}")
        print(f"   Std Dev: {std:.2f}")
        print(f"   Min: {min_val:.0f} | Max: {max_val:.0f}")

        last20 = vals.tail(20)
        if len(last20) > 0:
            print(f"\n📈 Last 20 Games:")
            print(f"   Mean: {last20.mean():.2f}")
            print(f"   Median: {last20.median():.2f}")
            diff = last20.mean() - mean
            print(f"   Difference from season: {diff:+.2f}")

        over = (vals > line).sum()
        under = (vals <= line).sum()
        print(f"\n🎯 Historical Performance vs Line {line}:")
        print(f"   Over: {over}/{len(vals)} ({100*over/len(vals):.1f}%)")
        print(f"   Under: {under}/{len(vals)} ({100*under/len(vals):.1f}%)")
        print("=" * 60)

    except Exception as e:
        print(f"[Debug] ⚠️ Debug projection failed: {e}")
        import traceback
        traceback.print_exc()


# Fix 3: Remove the broken recency weighting code
# Find this section in your analyze_single_prop() function (around line 1490):

    # ❌ DELETE THIS ENTIRE SECTION - it's causing the stat_col error
    # try:
    #     if len(df) >= 10:
    #         mean_l10 = df[stat_col].astype(float).tail(10).mean()
    #     else:
    #         mean_l10 = mean
    #     ...
    # except Exception as e:
    #     print(f"[Projection] ⚠️ Recency weighting failed: {e}")
    #     proj_stat = mean * context_mult

    # It's already calculated earlier in the function, so this duplicate is unnecessary
        
    

# ===============================
# ✅ ANALYZE SINGLE PROP — PropPulse+ v2025.3b (Universal Sanity-Calibrated Edition)
# ===============================
def analyze_single_prop(player, stat, line, odds, settings, debug_mode=False):
    """Analyze a single prop and return results dict - fully tuned + sanity-stabilized version."""

    import os
    import time
    import numpy as np
    import pandas as pd
    from scipy.stats import norm

    # ==========================
    # 🗂 Load player logs
    # ==========================
    data_path = settings.get("data_path", "data")
    os.makedirs(data_path, exist_ok=True)
    path = os.path.join(data_path, f"{player.replace(' ', '_')}.csv")

    need_refresh = not os.path.exists(path) or (time.time() - os.path.getmtime(path)) / 3600 > 24

    if need_refresh:
        print(f"[Data] ⏳ Refreshing logs for {player}...")
        try:
            df = fetch_player_data(player, settings=settings)
        except Exception as e:
            print(f"[BDL] ⚠️ BallDon'tLie failed: {e}")
            try:
                df = fetch_player_data(player, settings=settings, include_last_season=True)
            except Exception as e2:
                print(f"[Backup] ❌ Could not fetch any logs: {e2}")
                return None

        if df is None or len(df) == 0:
            print(f"[EXIT] ❌ No game logs for {player} — aborting analysis")
            return None

        # Cache fresh logs
        try:
            df.to_csv(path, index=False)
        except Exception:
            pass

    else:
        df = pd.read_csv(path)
        print(f"[Data] ✅ Loaded {len(df)} games for {player}")

    # ==========================
    # 🧼 Clean minutes / DNPs
    # ==========================
    if "MIN" in df.columns:
        def parse_minutes(val):
            if isinstance(val, str) and ":" in val:
                m, s = val.split(":")
                return int(m) + int(s) / 60
            try:
                return float(val)
            except Exception:
                return 0.0

        df["MIN"] = df["MIN"].apply(parse_minutes)
        df = df[df["MIN"] > 0]

    if df.empty:
        print(f"[EXIT] ❌ All games filtered out for {player} — aborting")
        return None

        # ==========================
    # 📊 Stat extraction
    # ==========================
    stat_norm = stat.replace(" ", "").upper()
    stat_col = STAT_MAP.get(stat_norm)

    if not stat_col:
        print(f"[Error] ❌ Stat '{stat}' not recognized (normalized '{stat_norm}')")
        return None

    if isinstance(stat_col, list):
        df["COMPOSITE"] = df[stat_col].sum(axis=1)
        vals = df["COMPOSITE"].astype(float)
        stat_key_for_recent = "COMPOSITE"
    else:
        if stat_col not in df.columns:
            print(f"[Error] ❌ Stat '{stat}' not found for {player}")
            return None
        vals = df[stat_col].astype(float)
        stat_key_for_recent = stat_col

    # ==========================
    # 📈 Core stats
    # ==========================
    n_games = len(vals)
    season_mean = vals.mean()
    std = vals.std() if n_games > 1 else 1.0
    mean_l10 = vals.tail(10).mean() if n_games >= 10 else season_mean
    mean_l20 = vals.tail(20).mean() if n_games >= 20 else season_mean

    # ==========================
    # 🔮 Baseline projection
    # ==========================
    recent_trend = (mean_l10 + mean_l20) / 2
    trend_weight = 0.15
    base_projection = (1 - trend_weight) * season_mean + trend_weight * recent_trend
    print(f"[Projection] Season={season_mean:.2f}, Recent trend={recent_trend:.2f} → base={base_projection:.2f}")

    # ==========================
    # ⏱ Minutes adjustment
    # ==========================
    if "MIN" in df.columns:
        season_mins = df["MIN"].mean()
        l10_mins = df["MIN"].tail(10).mean() if n_games >= 10 else season_mins
        mins_ratio = l10_mins / season_mins if season_mins > 0 else 1.0
        if abs(mins_ratio - 1.0) > 0.10:
            print(f"[Minutes] Season={season_mins:.1f}, L10={l10_mins:.1f} → ratio={mins_ratio:.3f}")
            base_projection *= mins_ratio

    # ==========================
    # 🎲 Empirical & normal probabilities
    # ==========================
    p_emp = float(np.mean(vals > line))
    p_norm = 1 - norm.cdf(line, season_mean, std if std > 0 else 1.0)
    p_base = 0.6 * p_norm + 0.4 * p_emp

    # ==========================
    # 🌍 Contextual factors
    # ==========================
    p_ha = get_homeaway_adjustment(player, stat, line, settings)
    p_l10 = get_recent_form(df, stat_key_for_recent, line)
    p_usage = get_usage_factor(player, stat, settings)

    # ==========================
    # 🧮 Opponent + DvP
    # ==========================
    opp = None
    team_abbr = None
    try:
        result = get_live_opponent_from_schedule(player, settings)
        if result and isinstance(result, tuple) and len(result) == 2:
            opp, team_abbr = result
        elif result:
            opp = result
            team_abbr = None
        if not opp:
            print(f"[Schedule] ⚠️ Could not determine opponent, using fallback...")
            opp = get_upcoming_opponent_abbr(player, settings)
            team_abbr = None
    except Exception as e:
        print(f"[Schedule] ❌ Opponent detection failed: {e}")
        try:
            opp = get_upcoming_opponent_abbr(player, settings)
        except Exception as e2:
            print(f"[Schedule] ❌ Fallback also failed: {e2}")
            opp = None

    pos = get_player_position_auto(player, df_logs=df, settings=settings)
    try:
        dvp_mult = get_dvp_multiplier(opp, pos, stat) if (opp and pos) else 1.0
    except Exception as e:
        print(f"[DvP] ⚠️ Could not apply DvP: {e}")
        dvp_mult = 1.0

    # ==========================
    # 📦 Probability stacking
    # ==========================
    maturity = min(1.0, n_games / 40)
    w_base = 0.40 + 0.15 * maturity
    w_l10 = 0.10 - 0.03 * maturity
    w_ha = 0.10
    w_dvp = 0.25 + 0.05 * maturity
    w_usage = 0.15 - 0.02 * maturity
    total = w_base + w_l10 + w_ha + w_dvp + w_usage
    w_base, w_l10, w_ha, w_dvp, w_usage = [w / total for w in (w_base, w_l10, w_ha, w_dvp, w_usage)]

    p_dvp = p_base * dvp_mult
    p_model_raw = (p_base * w_base +
                   p_l10 * w_l10 +
                   p_ha * w_ha +
                   p_dvp * w_dvp +
                   p_usage * w_usage)

    # ==========================
    # 🧠 Confidence system
    # ==========================
    base_conf = 1 - (std / season_mean) if season_mean > 0 else 0.5
    confidence = max(0.1, base_conf * maturity)

    if std > 0 and season_mean > 0:
        volatility_score = max(0.1, min(1.0, 1 - (std / season_mean)))
        confidence *= (0.7 + 0.3 * volatility_score)

    # Stat-specific adjustments, including new combos
    if stat.upper() in ["REB", "AST", "REB+AST"]:
        confidence *= 1.05
    elif stat.upper() in ["PTS", "PRA", "PTS+REB", "PTS+AST"]:
        confidence *= 0.95

    confidence = max(0.1, min(0.99, confidence))

    # ==========================
    # 🌡 Context multipliers
    # ==========================
    inj = get_injury_status(player, settings.get("injury_api_key"))
    team_total = get_team_total(player, settings)
    rest_days = get_rest_days(player, settings)

    team_mult = min(1.20, max(0.85, (team_total / 112) if team_total else 1.0))
    rest_mult = {0: 0.96, 1: 1.00, 2: 1.03}.get(rest_days, 1.05)
    dvp_mult_adjusted = max(0.80, min(1.25, dvp_mult))

    context_mult = dvp_mult_adjusted * team_mult * rest_mult
    print(f"[Context] DvP={dvp_mult_adjusted:.3f} × Team={team_mult:.3f} × Rest={rest_mult:.3f} = {context_mult:.3f}")

    # ==========================
    # 🎯 Apply context to projection
    # ==========================
    projection = base_projection * context_mult

    # ==========================
    # 🧱 Sanity check vs line
    # ==========================
    line_trust = 0.25
    deviation_ratio = projection / line if line > 0 else 1.0

    if deviation_ratio < 0.65:
        print(f"[Sanity] ⚠️ Projection too low ({projection:.1f} vs {line}) — blending 25% with line")
        projection = (1 - line_trust) * projection + line_trust * line
    elif deviation_ratio > 1.35:
        print(f"[Sanity] ⚠️ Projection too high ({projection:.1f} vs {line}) — blending 25% with line")
        projection = (1 - line_trust) * projection + line_trust * line

    proj_stat = float(projection)
    print(f"[Final] Projection={proj_stat:.2f} (base={base_projection:.2f} × context={context_mult:.3f})")

    # ==========================
    # 🚨 Deviation alert
    # ==========================
    deviation_pct = abs(proj_stat - line) / line * 100 if line > 0 else 0.0
    if deviation_pct > 25:
        print(f"[⚠️ ALERT] Projection deviates {deviation_pct:.1f}% from line!")
        print(f"   → Model: {proj_stat:.1f} | Line: {line}")
        print(f"   → This suggests missing context (injury news, role change, or vegas insider info)")
        confidence *= 0.70
        confidence = max(0.1, min(0.99, confidence))

    # ==========================
    # 💰 Probability & EV
    # ==========================
    p_model = max(0.05, min(p_model_raw * (0.5 + 0.5 * confidence), 0.95))
    p_book = american_to_prob(odds)
    ev_raw = ev_sportsbook(p_model, odds)
    ev = ev_raw * (0.5 + 0.5 * confidence)

    ev_cents = ev * 100.0
    ev_score = ev_cents * confidence

    # ==========================
    # 🏅 Dual grading systems
    # ==========================
    # Simple calibrated grade (assign_grade)
    try:
        grade_simple = assign_grade(ev_cents=ev_cents,
                                    confidence=confidence,
                                    model_prob=p_model,
                                    stat=stat)
    except Exception as e:
        print(f"[Grading-simple] ⚠️ Failed to apply assign_grade: {e}")
        grade_simple = "NEUTRAL"

    # Tuned model grade (grade_prop) using CONFIG_TUNED rules
    try:
        ev_pct = (p_model - p_book) * 100.0
        gap_abs = abs(proj_stat - line)
        grade_model = grade_prop(ev_pct, confidence, gap_abs, dvp_mult)
    except Exception as e:
        print(f"[Grading-model] ⚠️ Failed to apply grade_prop: {e}")
        grade_model = "NEUTRAL"

    # Primary grade used by the rest of the code
    grade = grade_model

    print(f"[EV] {player}: EV¢={ev_cents:+.2f} | Conf={confidence:.2f} | Score={ev_score:.2f} → Grade={grade} (simple={grade_simple})")

    # ==========================
    # 🧪 Debug panel
    # ==========================
    if debug_mode:
        try:
            debug_projection(df, stat=stat, line=line, player_name=player)
        except Exception as e:
            print(f"[Debug] ⚠️ Skipped debug projection: {e}")

    # ==========================
    # 📤 Final output dict
    # ==========================
    direction = "Higher" if proj_stat > line else "Lower"
    result_symbol = "⚠️" if abs(proj_stat - line) < 0.5 else ("✓" if direction == "Higher" else "✗")

    return {
        "player": player,
        "stat": stat,
        "line": float(line),
        "odds": int(odds),
        "projection": round(proj_stat, 2),
        "p_model": float(p_model),
        "p_book": float(p_book),
        "ev": float(ev),
        "n_games": int(n_games),
        "confidence": float(confidence),
        "grade": str(grade),             # Primary tuned grade (grade_prop)
        "grade_simple": str(grade_simple),  # Secondary simple grade (assign_grade)
        "opponent": opp or "N/A",
        "position": pos or "N/A",
        "dvp_mult": round(float(dvp_mult), 3),
        "injury": inj or "N/A",
        "direction": direction,
        "result": result_symbol
    }
# ================================================
# 🎯 GRADING LOGIC (using tuned config)
# ================================================
def grade_prop(ev_pct, conf, gap_abs, dvp):
    """
    Apply tuned rules from calibration to classify each prop.
    """
    cfg = CONFIG_TUNED
    if not cfg:
        return "NEUTRAL"

    filters = cfg["filters"]
    grading = cfg["grading"]
    dvp_rules = cfg["dvp_rules"]

    # --- Adjust confidence based on DvP ---
    if dvp and not np.isnan(dvp):
        if dvp < dvp_rules["penalty_threshold"]:
            conf *= dvp_rules["penalty_factor"]
        elif dvp > dvp_rules["boost_threshold"]:
            conf *= dvp_rules["boost_factor"]
        conf = max(0.0, min(1.0, conf))

    # --- Exclusions ---
    if gap_abs < filters["exclude_close_to_line_gap_abs"]:
        return "⚠️ TOO CLOSE"
    if conf < filters["exclude_low_confidence"]:
        return "LOW CONF"
    if ev_pct < filters["ev_floor_percent"]:
        return "LOW EV"

    # --- Grading tiers ---
    if (ev_pct >= grading["elite"]["ev_min_pct"]
        and conf >= grading["elite"]["conf_min"]
        and gap_abs >= grading["elite"]["gap_min"]):
        return "🔥 ELITE"

    if (ev_pct >= grading["solid"]["ev_min_pct"]
        and conf >= grading["solid"]["conf_min"]
        and gap_abs >= grading["solid"]["gap_min"]):
        return "✅ SOLID"

    return "NEUTRAL"
def batch_analyze_props(props_list, settings):
    """
    Runs analyze_single_prop() for a list of player props.
    Each entry in props_list should be a dict with:
        {'player': str, 'stat': str, 'line': float, 'odds': int}
    Returns a list of result dicts.
    """
    results = []
    for i, prop in enumerate(props_list, start=1):
        player = prop.get("player")
        stat = prop.get("stat")
        line = prop.get("line")
        odds = prop.get("odds")
        print(f"\n[{i}/{len(props_list)}] 📊 {player} — {stat} {line}")
        try:
            result = analyze_single_prop(player, stat, line, odds, settings, debug_mode=False)
            if result:
                results.append(result)
        except Exception as e:
            print(f"[Batch] ⚠️ Error analyzing {player}: {e}")
    return results

# ===============================
# 🏆 Display Summary Helper
# ===============================
def display_top_props(results, top_n=10):
    """Prints the top EV props in a clean summary format."""
    import pandas as pd

    if not results:
        print("⚠️ No results to display.")
        return

    df = pd.DataFrame(results)

    # Ensure numeric EV column
    try:
        df["EV¢"] = df["EV¢"].astype(float)
    except Exception:
        pass

    df_sorted = df.sort_values("EV¢", ascending=False).head(top_n)

    print("\n🏆 Top EV Props:")
    print("===============================================")
    for _, row in df_sorted.iterrows():
        print(f"{row['Player']} — {row['Stat']} {row['Line']}")
        print(f"   EV: {row['EV¢']}¢ | Conf: {row['Confidence']:.2f} | Grade: {row['Grade']}")
    print("===============================================")
# ===============================
# 🕓 DAILY SCHEDULE AUTO-REFRESH
# ===============================
import os, sys, json, time, requests
from datetime import datetime

def refresh_daily_schedule(settings):
    import pytz
    from datetime import datetime

    local_tz = pytz.timezone("America/Los_Angeles")
    today_real = datetime.now(local_tz).strftime("%Y-%m-%d")

    data_path = settings.get("data_path", "data/")
    os.makedirs(data_path, exist_ok=True)
    schedule_file = os.path.join(data_path, "schedule_today.json")

    print(f"[Schedule] 🔄 Refreshing schedule for {today_real}")

    try:
        url = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json"
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        full = r.json()

        games_today = []
        for gd in full["leagueSchedule"]["gameDates"]:
            if gd.get("gameDate") == today_real:
                games_today.extend(gd.get("games", []))

        # Save
        with open(schedule_file, "w") as f:
            json.dump({"date": today_real, "games": games_today}, f, indent=2)

        print(f"[Schedule] ✅ Saved {len(games_today)} games for {today_real}")
        return games_today

    except Exception as e:
        print(f"[Schedule] ❌ Failed: {e}")
        return []

# ===============================
# 🧠 MAIN (PropPulse+ v2025.4 — Stable)
# ===============================
def main():
    settings = load_settings()
    os.makedirs(settings.get("data_path", "data"), exist_ok=True)

    # ✅ ONLY keep this one:
    try:
        fetch_todays_slate_multiple_sources(settings)
    except Exception as e:
        print(f"[Startup] ⚠️ Schedule refresh failed: {e}")

    print("\n🧠 PropPulse+ Model v2025.4 — Player Prop EV Analyzer", flush=True)
    print("=====================================================\n", flush=True)

    while True:
        try:
            mode = input("Mode [1] Single  [2] Batch (manual)  [3] CSV file  [Q] Quit: ").strip().lower()
        except EOFError:
            print("\n(Input stream closed) Exiting.")
            return

        if mode in ("q", "quit", "exit"):
            print("Goodbye!")
            return

        # ---------- Single Prop ----------
        if mode in ("1", ""):
            try:
                player = input("Player name: ").strip()
                stat = input("Stat (PTS/REB/AST/REB+AST/PRA/FG3M): ").strip().upper()
                line = float(input("Line (e.g., 20.5): ").strip())
                odds = int(input("Odds (e.g., -110): ").strip())

                debug_input = input("Enable debug mode? (y/n, default=y): ").strip().lower()
                debug_mode = (debug_input != "n" and debug_input != "no")

                # Retry-safe analysis
                try:
                    result = analyze_single_prop(player, stat, line, odds, settings, debug_mode)
                except Exception as e:
                    print(f"[Retry] ⚠️ Initial analysis failed ({e}); retrying once...")
                    time.sleep(2)
                    result = analyze_single_prop(player, stat, line, odds, settings, debug_mode)

                if not result:
                    print("❌ Analysis returned no result.")
                else:
                    print("\n" + "=" * 60)
                    print(f"📊 {player} | {stat} Line {line}")
                    print(f"Games Analyzed: {result['n_games']}")
                    print(f"Model Projection: {result['projection']:.2f} {stat}")
                    print(f"Model Prob:  {result['p_model'] * 100:.1f}%")
                    print(f"Book Prob:   {result['p_book'] * 100:.1f}%")

                    ev_cents = result['ev'] * 100
                    edge_pct = (result['p_model'] - result['p_book']) * 100
                    conf = result.get('confidence', 0.0)
                    grade = result.get('grade', 'N/A')

                    print(f"EV: {ev_cents:+.1f}¢ per $1 | Edge: {edge_pct:+.2f}% | Confidence: {conf:.2f}")
                    print(f"Grade: {grade}")
                    print("🟢 Over Value" if result['projection'] > line else "🔴 Under Value")
                    print(f"Context → {result.get('position','N/A')} vs {result.get('opponent','N/A')} "
                          f"| DvP x{result.get('dvp_mult',1.0):.3f} | Injury: {result.get('injury','N/A')}")
                    print("=" * 60 + "\n")

                    # Compact summary
                    print(f"📈 {player} {stat} {line} → {grade} | EV={ev_cents:+.1f}¢ | Conf={conf:.2f}")

            except ValueError as ve:
                print(f"❌ Invalid input: {ve}")
            except Exception as e:
                print(f"❌ Error: {e}")

        elif mode == "2":
            print("Batch mode not implemented yet.")
        elif mode == "3":
            print("CSV mode not implemented yet.")
        else:
            print("Please choose 1, 2, 3, or Q.")
            input("\nPress Enter to continue...")


# ============================================================
# 🎯 PropPulse+ v2025.4 — Calibrated Grading Logic
# ============================================================
def assign_grade(ev_cents: float, confidence: float, model_prob: float, stat: str) -> str:
    """
    v2025.5 — Calibrated grading thresholds (based on 11/11 empirical hit rates)
    Tightens NEUTRAL range, boosts SOLID precision.
    """
    stat_weights = {
        "PTS":    {"ev": 7.0, "conf": 0.58},
        "REB":    {"ev": 5.0, "conf": 0.54},
        "AST":    {"ev": 5.0, "conf": 0.53},
        "PRA":    {"ev": 6.0, "conf": 0.55},
        "REB+AST": {"ev": 5.5, "conf": 0.54},
        "FG3M":   {"ev": 3.5, "conf": 0.50},
    }

    cfg = stat_weights.get(stat.upper(), {"ev": 5.0, "conf": 0.50})

    # 🔥 ELITE — top 5% confidence + edge
    if ev_cents >= cfg["ev"] * 1.8 and confidence >= (cfg["conf"] + 0.07) and model_prob >= 0.59:
        return "🔥 ELITE"

    # 💎 SOLID — balanced value with true edge
    elif ev_cents >= cfg["ev"] and confidence >= cfg["conf"] and model_prob >= 0.54:
        return "💎 SOLID"

    # ⚖️ NEUTRAL
    elif ev_cents >= 1.0 and model_prob >= 0.50 and confidence >= 0.40:
        return "⚖️ NEUTRAL"

    # 🚫 FADE
    else:
        return "🚫 FADE"


# ===============================
# Program entry point
# ===============================
if __name__ == "__main__":
    print("🧠 PropPulse+ Model v2025.4 — Player Prop EV Analyzer")
    print("=" * 60, flush=True)
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user. Exiting...")
    except Exception as e:
        import traceback
        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()
    finally:
        try:
            if sys.stdin.isatty():
                pass
            else:
                input("\nPress Enter to close...")
        except Exception:
            pass
def get_betting_grade(ev: float, model_prob: float, confidence: float) -> str:
    """
    Conservative betting filter based on 11/12 empirical results.
    Returns: STRONG_BET, LEAN, PASS
    """
    if ev >= 5.0 and model_prob >= 0.55:
        return "STRONG_BET"  # 63.6% hit rate historically
    
    elif ev >= 3.0 and model_prob >= 0.52:
        return "LEAN"  # Positive edge but smaller
    
    else:
        return "PASS"  # Insufficient edge
