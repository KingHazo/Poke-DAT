
import requests
import pandas as pd
import time

BASE_URL = "https://www.smogon.com/stats/2025-12/chaos/"
HEADERS  = {"User-Agent": "ShowdownStatsScraper/1.0 (educational project)"}

USAGE_CSV    = "pokemon_usage.csv"
TEAMMATE_CSV = "pokemon_teammates.csv"

#Gen 9 tiers with their appropriate rating cutoff
GEN9_TIERS = [
    ("gen9ubers",      "Ubers",       1695),
    ("gen9ou",         "OU",          1695),
    ("gen9uu",         "UU",          1695),
    ("gen9ru",         "RU",          1695),
    ("gen9nu",         "NU",          1695),
    ("gen9pu",         "PU",          1695),
    ("gen9lc",         "LC",          1695),
    ("gen9doublesou",  "Doubles OU",  1695),
    ("gen9nationaldex","National Dex",1695),
    ("gen9anythinggoes","AG",         1695),
    ("gen9monotype",   "Monotype",    1695),
]


def fetch_tier(format_id, cutoff):
    """Fetch chaos JSON for a given format and cutoff. Returns None if not found."""
    url = f"{BASE_URL}{format_id}-{cutoff}.json"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code == 404:
            #Try 1500 cutoff as fallback
            url2 = f"{BASE_URL}{format_id}-1500.json"
            resp = requests.get(url2, headers=HEADERS, timeout=30)
            if resp.status_code == 404:
                return None, url
        resp.raise_for_status()
        return resp.json(), url
    except Exception as e:
        print(f"  ERROR fetching {url}: {e}")
        return None, url


def parse_usage(data, tier_label):
    rows = []
    for name, pdata in data.get("data", {}).items():
        usage_pct = pdata.get("usage", 0) * 100
        raw_count = pdata.get("Raw count", 0)
        rows.append({
            "pokemon":   name,
            "tier":      tier_label,
            "usage_pct": round(usage_pct, 4),
            "raw_count": raw_count,
        })
    df = pd.DataFrame(rows).sort_values("usage_pct", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def parse_teammates(data, tier_label):
    rows = []
    for name, pdata in data.get("data", {}).items():
        for teammate, score in pdata.get("Teammates", {}).items():
            rows.append({
                "pokemon":  name,
                "tier":     tier_label,
                "teammate": teammate,
                "score":    round(score * 100, 4),
            })
    return pd.DataFrame(rows)


def main():
    all_usage    = []
    all_teammates = []

    for format_id, tier_label, cutoff in GEN9_TIERS:
        print(f"Fetching {tier_label} ({format_id}-{cutoff})...", end=" ", flush=True)
        data, url = fetch_tier(format_id, cutoff)
        if data is None:
            print("NOT FOUND — skipping")
            continue

        n = len(data.get("data", {}))
        battles = data.get("info", {}).get("number of battles", "?")
        print(f"{n} Pokemon, {battles:,} battles" if isinstance(battles, int) else f"{n} Pokemon")

        all_usage.append(parse_usage(data, tier_label))
        all_teammates.append(parse_teammates(data, tier_label))
        time.sleep(0.3)

    if not all_usage:
        print("No data fetched — check your internet connection.")
        return

    usage_df    = pd.concat(all_usage,     ignore_index=True)
    teammate_df = pd.concat(all_teammates, ignore_index=True)

    usage_df.to_csv(USAGE_CSV, index=False)
    teammate_df.to_csv(TEAMMATE_CSV, index=False)

    print(f"\nSaved {USAGE_CSV}    — {len(usage_df):,} rows")
    print(f"Saved {TEAMMATE_CSV} — {len(teammate_df):,} rows")

    print("\nPokémon per tier:")
    print(usage_df.groupby("tier").size().to_string())

    print("\nTop 5 in each tier:")
    for tier, grp in usage_df.groupby("tier"):
        top = grp.nsmallest(5, "rank")[["rank","pokemon","usage_pct"]]
        print(f"\n  [{tier}]")
        print(top.to_string(index=False))


if __name__ == "__main__":
    main()
