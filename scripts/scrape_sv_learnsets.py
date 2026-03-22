
import re
import time
import requests
import pandas as pd
from tqdm import tqdm

POKEMON_CSV = 'pokemon.csv'
OUTPUT_CSV  = 'pokemon_sv_learnsets.csv'
DELAY       = 0.3
MAX_RETRIES = 3
HEADERS     = {'User-Agent': 'SVLearnsetScraper/1.0 (educational project)'}
BASE_URL    = 'https://pokeapi.co/api/v2/pokemon'

#Newest to oldest — first match wins for each Pokemon
VERSION_PRIORITY = [
    'the-indigo-disk',
    'the-teal-mask',
    'scarlet-violet',
    'legends-arceus',
    'brilliant-diamond-shining-pearl',
    'sword-shield',
    'the-crown-tundra',
    'isle-of-armor',
    'ultra-sun-ultra-moon',
    'sun-moon',
    'omega-ruby-alpha-sapphire',
    'x-y',
    'black-2-white-2',
    'black-white',
    'heartgold-soulsilver',
    'platinum',
    'diamond-pearl',
    'firered-leafgreen',
    'emerald',
    'ruby-sapphire',
    'crystal',
    'gold-silver',
    'yellow',
    'red-blue',
]

SLUG_OVERRIDES = {
    #Special characters
    "Nidoran\u2640":                "nidoran-f",
    "Nidoran\u2642":                "nidoran-m",
    "Nidoran Female":                "nidoran-f",
    "Nidoran Male":                  "nidoran-m",
    "Mr. Mime":                      "mr-mime",
    "Mime Jr.":                      "mime-jr",
    "Mr. Rime":                      "mr-rime",
    "Farfetch'd":                   "farfetchd",
    "Galarian Farfetch'd":          "farfetchd-galar",
    "Sirfetch'd":                   "sirfetchd",
    "Flab\u00e9b\u00e9":           "flabebe",
    "Jangmo-o":                      "jangmo-o",
    "Hakamo-o":                      "hakamo-o",
    "Kommo-o":                       "kommo-o",
    "Galarian Slowpoke":             "slowpoke-galar",
    "Galarian Slowbro":              "slowbro-galar",
    "Galarian Slowking":             "slowking-galar",
    "Tapu Koko":                     "tapu-koko",
    "Tapu Lele":                     "tapu-lele",
    "Tapu Bulu":                     "tapu-bulu",
    "Tapu Fini":                     "tapu-fini",
    "Type: Null":                    "type-null",
    "Porygon-Z":                     "porygon-z",
    "Ho-Oh":                         "ho-oh",
    "Chi-Yu":                        "chi-yu",
    "Ting-Lu":                       "ting-lu",
    "Chien-Pao":                     "chien-pao",
    "Wo-Chien":                      "wo-chien",
    #Paradox Pokemon
    "Roaring Moon":                  "roaring-moon",
    "Iron Valiant":                  "iron-valiant",
    "Iron Treads":                   "iron-treads",
    "Iron Bundle":                   "iron-bundle",
    "Iron Hands":                    "iron-hands",
    "Iron Jugulis":                  "iron-jugulis",
    "Iron Moth":                     "iron-moth",
    "Iron Thorns":                   "iron-thorns",
    "Flutter Mane":                  "flutter-mane",
    "Slither Wing":                  "slither-wing",
    "Sandy Shocks":                  "sandy-shocks",
    "Scream Tail":                   "scream-tail",
    "Brute Bonnet":                  "brute-bonnet",
    "Walking Wake":                  "walking-wake",
    "Gouging Fire":                  "gouging-fire",
    "Raging Bolt":                   "raging-bolt",
    "Iron Crown":                    "iron-crown",
    "Iron Boulder":                  "iron-boulder",
    "Iron Leaves":                   "iron-leaves",
    #Ursaluna
    "Bloodmoon Ursaluna":            "ursaluna-bloodmoon",
    "Ursaluna\nBloodmoon":          "ursaluna-bloodmoon",
    #Gender-split forms (female variants)
    "Oinkologne Female":             "oinkologne-female",
    "Indeedee Female":               "indeedee-female",
    "Meowstic Female":               "meowstic-female",
    "Basculegion Female":            "basculegion-female",
    #Gender-split forms (male = base)
    "Meowstic\nMale":               "meowstic",
    "Indeedee\nMale":               "indeedee",
    "Basculegion\nMale":            "basculegion",
    "Oinkologne\nMale":             "oinkologne",
    "Frillish":                      "frillish",
    "Jellicent":                     "jellicent",
    "Pyroar":                        "pyroar",
    #Partner forms
    "Partner Pikachu":               "pikachu-partner",
    "Partner Eevee":                 "eevee-partner",
    #Tauros Paldean breeds
    "Paldean Tauros":                "tauros-paldea-combat-breed",
    "Tauros\nCombat Breed":         "tauros-paldea-combat-breed",
    "Tauros\nBlaze Breed":          "tauros-paldea-blaze-breed",
    "Tauros\nAqua Breed":           "tauros-paldea-aqua-breed",
    #Castform
    "Castform\nSunny Form":         "castform-sunny",
    "Castform\nRainy Form":         "castform-rainy",
    "Castform\nSnowy Form":         "castform-snowy",
    #Deoxys
    "Deoxys\nNormal Forme":         "deoxys-normal",
    "Deoxys\nAttack Forme":         "deoxys-attack",
    "Deoxys\nDefense Forme":        "deoxys-defense",
    "Deoxys\nSpeed Forme":          "deoxys-speed",
    #Burmy / Wormadam
    "Burmy\nPlant Cloak":           "burmy-plant",
    "Burmy\nSandy Cloak":           "burmy-sandy",
    "Burmy\nTrash Cloak":           "burmy-trash",
    "Wormadam\nPlant Cloak":        "wormadam-plant",
    "Wormadam\nSandy Cloak":        "wormadam-sandy",
    "Wormadam\nTrash Cloak":        "wormadam-trash",
    #Rotom
    "Rotom\nHeat Rotom":            "rotom-heat",
    "Rotom\nWash Rotom":            "rotom-wash",
    "Rotom\nFrost Rotom":           "rotom-frost",
    "Rotom\nFan Rotom":             "rotom-fan",
    "Rotom\nMow Rotom":             "rotom-mow",
    #Sinnoh legends
    "Dialga\nOrigin Forme":         "dialga-origin",
    "Palkia\nOrigin Forme":         "palkia-origin",
    "Giratina\nAltered Forme":      "giratina-altered",
    "Giratina\nOrigin Forme":       "giratina-origin",
    "Shaymin\nLand Forme":          "shaymin-land",
    "Shaymin\nSky Forme":           "shaymin-sky",
    #Basculin
    "Basculin\nRed-Striped Form":   "basculin-red-striped",
    "Basculin\nBlue-Striped Form":  "basculin-blue-striped",
    "Basculin\nWhite-Striped Form": "basculin-white-striped",
    #Darmanitan
    "Darmanitan\nStandard Mode":    "darmanitan-standard",
    "Darmanitan\nZen Mode":         "darmanitan-zen",
    "Darmanitan\nGalarian Standard Mode": "darmanitan-galar-standard",
    "Darmanitan\nGalarian Zen Mode":"darmanitan-galar-zen",
    #Forces of Nature / Genies
    "Tornadus\nIncarnate Forme":    "tornadus-incarnate",
    "Tornadus\nTherian Forme":      "tornadus-therian",
    "Thundurus\nIncarnate Forme":   "thundurus-incarnate",
    "Thundurus\nTherian Forme":     "thundurus-therian",
    "Landorus\nIncarnate Forme":    "landorus-incarnate",
    "Landorus\nTherian Forme":      "landorus-therian",
    "Enamorus\nIncarnate Forme":    "enamorus-incarnate",
    "Enamorus\nTherian Forme":      "enamorus-therian",
    #Kyurem
    "White Kyurem":                  "kyurem-white",
    "Black Kyurem":                  "kyurem-black",
    #Keldeo / Meloetta
    "Keldeo\nOrdinary Form":        "keldeo-ordinary",
    "Keldeo\nResolute Form":        "keldeo-resolute",
    "Meloetta\nAria Forme":         "meloetta-aria",
    "Meloetta\nPirouette Forme":    "meloetta-pirouette",
    #Greninja
    "Ash-Greninja":                  "greninja-ash",
    #Aegislash
    "Aegislash\nShield Forme":      "aegislash-shield",
    "Aegislash\nBlade Forme":       "aegislash-blade",
    #Pumpkaboo / Gourgeist sizes
    "Pumpkaboo\nAverage Size":      "pumpkaboo-average",
    "Pumpkaboo\nSmall Size":        "pumpkaboo-small",
    "Pumpkaboo\nLarge Size":        "pumpkaboo-large",
    "Pumpkaboo\nSuper Size":        "pumpkaboo-super",
    "Gourgeist\nAverage Size":      "gourgeist-average",
    "Gourgeist\nSmall Size":        "gourgeist-small",
    "Gourgeist\nLarge Size":        "gourgeist-large",
    "Gourgeist\nSuper Size":        "gourgeist-super",
    #Zygarde
    "Zygarde\n50% Forme":           "zygarde-50",
    "Zygarde\n10% Forme":           "zygarde-10",
    "Zygarde\nComplete Forme":      "zygarde-complete",
    #Hoopa
    "Hoopa\nHoopa Confined":        "hoopa",
    "Hoopa\nHoopa Unbound":         "hoopa-unbound",
    #Oricorio
    "Oricorio\nBaile Style":        "oricorio-baile",
    "Oricorio\nPom-Pom Style":      "oricorio-pom-pom",
    "Oricorio\nPa'u Style":        "oricorio-pau",
    "Oricorio\nSensu Style":        "oricorio-sensu",
    #Rockruff / Lycanroc
    "Rockruff\nOwn Tempo Rockruff": "rockruff-own-tempo",
    "Lycanroc\nMidday Form":        "lycanroc-midday",
    "Lycanroc\nMidnight Form":      "lycanroc-midnight",
    "Lycanroc\nDusk Form":          "lycanroc-dusk",
    #Wishiwashi
    "Wishiwashi\nSolo Form":        "wishiwashi-solo",
    "Wishiwashi\nSchool Form":      "wishiwashi-school",
    #Minior
    "Minior\nMeteor Form":          "minior-red-meteor",
    "Minior\nCore Form":            "minior-red",
    #Toxtricity
    "Toxtricity\nAmped Form":       "toxtricity-amped",
    "Toxtricity\nLow Key Form":     "toxtricity-low-key",
    #Eiscue
    "Eiscue\nIce Face":             "eiscue-ice",
    "Eiscue\nNoice Face":           "eiscue-noice",
    #Morpeko
    "Morpeko\nFull Belly Mode":     "morpeko-full-belly",
    "Morpeko\nHangry Mode":         "morpeko-hangry",
    #Zacian / Zamazenta / Eternatus
    "Zacian\nHero of Many Battles": "zacian-hero",
    "Zacian\nCrowned Sword":        "zacian-crowned",
    "Zamazenta\nHero of Many Battles": "zamazenta-hero",
    "Zamazenta\nCrowned Shield":    "zamazenta-crowned",
    "Eternatus\nEternamax":         "eternatus-eternamax",
    #Urshifu
    "Urshifu\nSingle Strike Style": "urshifu-single-strike",
    "Urshifu\nRapid Strike Style":  "urshifu-rapid-strike",
    #Calyrex
    "Calyrex\nIce Rider":           "calyrex-ice",
    "Calyrex\nShadow Rider":        "calyrex-shadow",
    #Necrozma
    "Dusk Mane Necrozma":            "necrozma-dusk-mane",
    "Dawn Wings Necrozma":           "necrozma-dawn-wings",
    "Ultra Necrozma":                "necrozma-ultra",
    #Maushold
    "Maushold\nFamily of Four":     "maushold-family-of-four",
    "Maushold\nFamily of Three":    "maushold-family-of-three",
    #Squawkabilly
    "Squawkabilly\nGreen Plumage":  "squawkabilly-green-plumage",
    "Squawkabilly\nBlue Plumage":   "squawkabilly-blue-plumage",
    "Squawkabilly\nYellow Plumage": "squawkabilly-yellow-plumage",
    "Squawkabilly\nWhite Plumage":  "squawkabilly-white-plumage",
    #Palafin
    "Palafin\nZero Form":           "palafin-zero",
    "Palafin\nZero Form":           "palafin",
    "Palafin\nHero Form":           "palafin-hero",
    #Tatsugiri
    "Tatsugiri\nCurly Form":        "tatsugiri-curly",
    "Tatsugiri\nDroopy Form":       "tatsugiri-droopy",
    "Tatsugiri\nStretchy Form":     "tatsugiri-stretchy",
    #Dudunsparce
    "Dudunsparce\nTwo-Segment Form":   "dudunsparce-two-segment",
    "Dudunsparce\nThree-Segment Form": "dudunsparce-three-segment",
    #Gimmighoul
    "Gimmighoul\nChest Form":       "gimmighoul",
    "Gimmighoul\nRoaming Form":     "gimmighoul-roaming",
    #Ogerpon
    "Ogerpon\nTeal Mask":           "ogerpon-teal-mask",
    "Ogerpon\nWellspring Mask":     "ogerpon-wellspring-mask",
    "Ogerpon\nHearthflame Mask":    "ogerpon-hearthflame-mask",
    "Ogerpon\nCornerstone Mask":    "ogerpon-cornerstone-mask",
    #Terapagos
    "Terapagos\nNormal Form":       "terapagos",
    "Terapagos\nTerastal Form":     "terapagos-terastal",
    "Terapagos\nStellar Form":      "terapagos-stellar",
    #Gender forms PokéAPI stores with explicit -male suffix
    "Frillish":                      "frillish-male",
    "Jellicent":                     "jellicent-male",
    "Pyroar":                        "pyroar-male",
    "Meowstic\nMale":               "meowstic-male",
    "Indeedee\nMale":               "indeedee-male",
    "Basculegion\nMale":            "basculegion-male",
    "Oinkologne\nMale":             "oinkologne-male",
    #Mimikyu - PokéAPI uses the form suffix
    "Mimikyu":                       "mimikyu-disguised",
}


def name_to_slug(csv_name):
    name = str(csv_name).strip()
    #Check the FULL name (including \n) against overrides first
    if name in SLUG_OVERRIDES:
        return SLUG_OVERRIDES[name]
    #Then strip to the variant part for further processing
    if '\n' in name:
        name = name.split('\n')[1].strip()
    if name in SLUG_OVERRIDES:
        return SLUG_OVERRIDES[name]
    for prefix, suffix in [
        ('Alolan ',   '-alola'),
        ('Galarian ',  '-galar'),
        ('Hisuian ',   '-hisui'),
        ('Paldean ',   '-paldea'),
    ]:
        if name.startswith(prefix):
            base = re.sub(r"['\.]", '', name[len(prefix):]).lower().replace(' ', '-')
            return base + suffix
    if name.startswith('Mega '):
        rest = name[5:]
        if rest.endswith(' X'):
            return rest[:-2].lower().replace(' ', '-') + '-mega-x'
        if rest.endswith(' Y'):
            return rest[:-2].lower().replace(' ', '-') + '-mega-y'
        return rest.lower().replace(' ', '-') + '-mega'
    if name.startswith('Primal '):
        return name[7:].lower().replace(' ', '-') + '-primal'
    if name.startswith('Gigantamax '):
        return name[11:].lower().replace(' ', '-') + '-gmax'
    slug = name.lower()
    slug = slug.replace('. ', '-').replace('.', '').replace("'", '').replace(' ', '-')
    slug = re.sub(r'[^a-z0-9\-]', '', slug)
    return re.sub(r'-+', '-', slug).strip('-')


def _try_fetch(slug):
    """Fetch PokéAPI JSON for slug. Returns None on 404 or error."""
    url = f"{BASE_URL}/{slug}"
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt == MAX_RETRIES - 1:
                print(f"\n  WARNING: failed to fetch {slug}: {e}")
                return None
            time.sleep(1)
    return None


def fetch_moves(slug):
    #Try the exact slug first, then fall back by progressively stripping the last hyphen-segment
    parts = slug.split('-')
    candidates = [slug] + ['-'.join(parts[:i]) for i in range(len(parts)-1, 0, -1)]

    data = None
    for candidate in candidates:
        data = _try_fetch(candidate)
        if data is not None:
            break

    if data is None:
        return [], ''

    #Bucket moves by version group
    vg_moves = {}
    for move_entry in data.get('moves', []):
        move_name = move_entry['move']['name'].replace('-', ' ').title()
        for vgd in move_entry.get('version_group_details', []):
            vg     = vgd['version_group']['name']
            method = vgd['move_learn_method']['name']
            level  = vgd.get('level_learned_at', 0)
            if vg not in vg_moves:
                vg_moves[vg] = []
            vg_moves[vg].append((move_name, method, level))

    #Return data from the highest-priority version group that has moves
    for vg in VERSION_PRIORITY:
        if vg in vg_moves and vg_moves[vg]:
            seen, rows = set(), []
            for move_name, method, level in vg_moves[vg]:
                key = (move_name, method)
                if key not in seen:
                    seen.add(key)
                    rows.append({'move_name': move_name, 'learn_method': method, 'level': level})
            return rows, vg

    #Fallback: use whichever version group has the most moves
    if vg_moves:
        best_vg = max(vg_moves, key=lambda v: len(vg_moves[v]))
        seen, rows = set(), []
        for move_name, method, level in vg_moves[best_vg]:
            key = (move_name, method)
            if key not in seen:
                seen.add(key)
                rows.append({'move_name': move_name, 'learn_method': method, 'level': level})
        return rows, best_vg

    return [], ''


def main():
    pokemon_df = pd.read_csv(POKEMON_CSV)
    names = pokemon_df['Name'].tolist()

    print(f"Scraping learnsets for {len(names)} Pokemon rows...")

    all_rows  = []
    skipped   = []
    vg_counts = {}

    for csv_name in tqdm(names, unit='mon'):
        slug = name_to_slug(csv_name)
        moves, vg_used = fetch_moves(slug)

        if not moves:
            skipped.append((csv_name, slug))
        else:
            vg_counts[vg_used] = vg_counts.get(vg_used, 0) + 1
            for m in moves:
                all_rows.append({
                    'pokemon_name':  csv_name,
                    'move_name':     m['move_name'],
                    'learn_method':  m['learn_method'],
                    'level':         m['level'],
                    'version_group': vg_used,
                })

        time.sleep(DELAY)

    df = pd.DataFrame(all_rows, columns=[
        'pokemon_name', 'move_name', 'learn_method', 'level', 'version_group'
    ])
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')

    print(f"\nDone")
    print(f"  Total learnset rows : {len(df):,}")
    print(f"  Unique Pokemon      : {df['pokemon_name'].nunique()}")
    print(f"  Unique moves        : {df['move_name'].nunique()}")
    print(f"  Saved to            : {OUTPUT_CSV}")
    print(f"\nVersion groups used:")
    for vg, count in sorted(vg_counts.items(), key=lambda x: -x[1]):
        print(f"  {vg:45s} {count} Pokemon")

    if skipped:
        print(f"\nSkipped ({len(skipped)} - not found in PokéAPI):")
        for csv_name, slug in skipped:
            print(f"  {str(csv_name)!r:40s} -> {slug}")


if __name__ == '__main__':
    main()
