"""
scrape_encounters.py
--------------------
Downloads the relevant CSV tables from the PokeAPI GitHub repository and
joins them into a single flat encounter table saved as pokemon_encounters.csv.

COVERAGE NOTE: PokeAPI encounter data is complete for Generations 1-6
(Red/Blue through X/Y). Generation 7+ location data is not populated.

Output columns:
    pokemon_name, pokemon_id, type_1, type_2,
    location, location_area, version, version_group, generation,
    encounter_method, min_level, max_level

Usage:
    pip install requests pandas
    python scrape_encounters.py
"""

import requests
import pandas as pd
import io
import time

BASE_URL = "https://raw.githubusercontent.com/PokeAPI/pokeapi/master/data/v2/csv/"
HEADERS  = {"User-Agent": "EncounterScraper/1.0 (educational project)"}
OUTPUT   = "pokemon_encounters.csv"

TABLES = [
    "encounters",
    "encounter_slots",
    "encounter_methods",
    "location_areas",
    "locations",
    "location_names",
    "versions",
    "version_groups",
    "pokemon",
    "pokemon_types",
    "types",
    "type_names",
]


def fetch_csv(table):
    url = BASE_URL + table + ".csv"
    print(f"  Fetching {table}.csv ...", end=" ", flush=True)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    print(f"{len(df):,} rows  |  cols: {df.columns.tolist()}")
    time.sleep(0.2)
    return df


def main():
    print("Downloading PokeAPI CSV tables from GitHub...\n")
    tables = {}
    for t in TABLES:
        try:
            tables[t] = fetch_csv(t)
        except Exception as e:
            print(f"  FAILED: {e}")
            tables[t] = pd.DataFrame()

    enc       = tables["encounters"]
    slots     = tables["encounter_slots"]
    methods   = tables["encounter_methods"]
    areas     = tables["location_areas"]
    locations = tables["locations"]
    loc_names = tables["location_names"]
    versions  = tables["versions"]
    vgroups   = tables["version_groups"]
    pokemon   = tables["pokemon"]
    ptypes    = tables["pokemon_types"]
    types_df  = tables["types"]
    tnames    = tables["type_names"]

    print("\nColumn inspection:")
    for name, df in [("encounters", enc), ("encounter_slots", slots),
                     ("versions", versions), ("version_groups", vgroups)]:
        print(f"  {name}: {df.columns.tolist()}")

    print("\nJoining tables...")

    # ── Type names (English = language id 9) ─────────────────────────────────
    if "local_language_id" in tnames.columns:
        type_en = tnames[tnames["local_language_id"] == 9][["type_id","name"]]
    else:
        type_en = tnames[["type_id","name"]]
    type_en = type_en.rename(columns={"name":"type_name"})

    pt1 = (ptypes[ptypes["slot"] == 1]
           .merge(type_en, on="type_id")[["pokemon_id","type_name"]]
           .rename(columns={"type_name":"type_1"}))
    pt2 = (ptypes[ptypes["slot"] == 2]
           .merge(type_en, on="type_id")[["pokemon_id","type_name"]]
           .rename(columns={"type_name":"type_2"}))

    # ── Base Pokémon (default forms only) ─────────────────────────────────────
    poke_base = (pokemon[pokemon["is_default"] == 1]
                 [["id","identifier"]]
                 .rename(columns={"id":"pokemon_id","identifier":"pokemon_slug"}))

    # ── Location names (English) ──────────────────────────────────────────────
    ln_en = (loc_names[loc_names["local_language_id"] == 9]
             [["location_id","name"]]
             .rename(columns={"name":"location_name"}))

    # ── Encounter method names ─────────────────────────────────────────────────
    # encounter_methods has columns: id, identifier, ...
    method_map = (methods[["id","identifier"]]
                  .rename(columns={"id":"encounter_method_id",
                                   "identifier":"encounter_method"}))

    # ── Encounter slots → method id only (avoid version_group_id collision) ───
    # We get version_group via versions table instead
    slots_methods = (slots[["id","encounter_method_id"]]
                     .rename(columns={"id":"encounter_slot_id"}))

    # ── Versions → version_group_id ───────────────────────────────────────────
    ver_cols = ["id","identifier","version_group_id"]
    ver_vg = (versions[[c for c in ver_cols if c in versions.columns]]
              .rename(columns={"id":"version_id","identifier":"version_name"}))

    # ── Version groups → generation_id ────────────────────────────────────────
    vg_cols = ["id","identifier","generation_id"]
    vg_info = (vgroups[[c for c in vg_cols if c in vgroups.columns]]
               .rename(columns={"id":"version_group_id",
                                 "identifier":"version_group_name"}))

    gen_map = {
        1: "generation-i",    2: "generation-ii",   3: "generation-iii",
        4: "generation-iv",   5: "generation-v",    6: "generation-vi",
        7: "generation-vii",  8: "generation-viii", 9: "generation-ix",
    }

    # ── Location areas ────────────────────────────────────────────────────────
    area_cols = ["id","identifier","location_id"]
    areas_slim = (areas[[c for c in area_cols if c in areas.columns]]
                  .rename(columns={"id":"location_area_id",
                                    "identifier":"location_area_slug"}))

    # ── Assemble ──────────────────────────────────────────────────────────────
    df = enc.copy()
    print(f"  Starting rows: {len(df):,}")

    # 1. Add encounter method (from slots, method id only)
    df = df.merge(slots_methods, on="encounter_slot_id", how="left")

    # 2. Resolve method name
    df = df.merge(method_map, on="encounter_method_id", how="left")

    # 3. Add version info (brings version_group_id cleanly)
    df = df.merge(ver_vg, on="version_id", how="left")

    # 4. Add version group info (generation etc.)
    if "version_group_id" in df.columns and "version_group_id" in vg_info.columns:
        df = df.merge(vg_info, on="version_group_id", how="left")
    else:
        print("  WARNING: version_group_id missing, skipping vg_info merge")
        df["version_group_name"] = ""
        df["generation_id"]      = None

    df["generation"] = df["generation_id"].map(gen_map)

    # 5. Location area → location
    df = df.merge(areas_slim, on="location_area_id", how="left")
    df = df.merge(ln_en,      on="location_id",      how="left")
    df["location_name"] = df["location_name"].fillna(df.get("location_area_slug",""))

    # 6. Pokémon name and types
    df = df.merge(poke_base, on="pokemon_id", how="left")
    df = df.merge(pt1,       on="pokemon_id", how="left")
    df = df.merge(pt2,       on="pokemon_id", how="left")
    df["type_2"] = df.get("type_2", pd.Series([""] * len(df))).fillna("")

    df["pokemon_name"] = (df["pokemon_slug"]
                          .str.replace("-", " ", regex=False)
                          .str.title())

    # ── Final columns ─────────────────────────────────────────────────────────
    keep = {
        "pokemon_name": "pokemon_name",
        "pokemon_id":   "pokemon_id",
        "type_1":       "type_1",
        "type_2":       "type_2",
        "location_name":"location",
        "location_area_slug": "location_area",
        "version_name": "version",
        "version_group_name": "version_group",
        "generation":   "generation",
        "encounter_method": "encounter_method",
        "min_level":    "min_level",
        "max_level":    "max_level",
    }
    available = {k: v for k, v in keep.items() if k in df.columns}
    final = df[list(available.keys())].rename(columns=available)
    final = final.dropna(subset=["pokemon_name","location"]).reset_index(drop=True)

    final.to_csv(OUTPUT, index=False)

    print(f"\nSaved: {OUTPUT}")
    print(f"  Rows:             {len(final):,}")
    print(f"  Unique Pokemon:   {final['pokemon_name'].nunique()}")
    print(f"  Unique locations: {final['location'].nunique()}")
    if "generation" in final.columns:
        print(f"\n  Rows per generation:")
        print(final.groupby("generation").size().to_string())
    if "version" in final.columns:
        print(f"\n  Versions: {sorted(final['version'].dropna().unique().tolist())}")


if __name__ == "__main__":
    main()
