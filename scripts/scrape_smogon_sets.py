
import requests
import json

OUTPUT_FILE = 'pokemon_smogon_sets.json'
BASE_URL    = 'https://data.pkmn.cc/sets/'
HEADERS     = {'User-Agent': 'SmogonSetsScraper/1.0 (educational project)'}

#Tiers in priority order — if a Pokémon appears in multiple tiers, the highest-priority tier's sets are kept (but all sets are merged).
GEN9_TIERS = [
    ('gen9ubers.json',      'Ubers'),
    ('gen9ou.json',         'OU'),
    ('gen9uu.json',         'UU'),
    ('gen9ru.json',         'RU'),
    ('gen9nu.json',         'NU'),
    ('gen9pu.json',         'PU'),
    ('gen9lc.json',         'LC'),
    ('gen9doublesou.json',  'Doubles OU'),
    ('gen9nationaldex.json','National Dex'),
    ('gen9anythinggoes.json','AG'),
]


def fetch_tier(filename):
    url = BASE_URL + filename
    resp = requests.get(url, headers=HEADERS, timeout=20)
    if resp.status_code == 404:
        print(f"  {filename} — not found, skipping")
        return {}
    resp.raise_for_status()
    return resp.json()


def main():
    combined = {}   #pokemon_name -> {tier, sets}

    for filename, tier_label in GEN9_TIERS:
        print(f"Fetching {tier_label} ({filename})...")
        data = fetch_tier(filename)
        added = 0
        for pokemon_name, sets in data.items():
            if pokemon_name not in combined:
                combined[pokemon_name] = {'tier': tier_label, 'sets': sets}
                added += 1
            else:
                #Merge sets from lower tiers (adds alternate formats)
                for set_name, set_data in sets.items():
                    key = f"{set_name} [{tier_label}]"
                    combined[pokemon_name]['sets'][key] = set_data
        print(f"  Added {added} new Pokémon, {len(data)} total in tier")

    print(f"\nTotal Pokémon with sets: {len(combined)}")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2)

    print(f"Saved to {OUTPUT_FILE}")

    #Print a few samples to check it worked
    sample_names = list(combined.keys())[:3]
    for name in sample_names:
        print(f"\n--- {name} ({combined[name]['tier']}) ---")
        for set_name, s in combined[name]['sets'].items():
            moves = s.get('moves', [])
            moves_flat = [
                m if isinstance(m, str) else ' / '.join(m)
                for m in moves
            ]
            print(f"  [{set_name}] {moves_flat}")


if __name__ == '__main__':
    main()
