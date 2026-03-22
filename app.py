import streamlit as st
import pandas as pd
import anthropic as _anthropic
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LinearSegmentedColormap

#Page Config
st.set_page_config(page_title="Poke-DAT", layout="wide")

#The Pokedex Theme
def get_pokedex_colors(n):
    #Defining a gradient based on Pokedex colors
    colors_list = ['#EF5350', '#EC407A', '#AB47BC', '#7E57C2', '#5C6BC0', '#42A5F5']
    cmap = LinearSegmentedColormap.from_list('pokedex', colors_list)
    return [cmap(i/n) for i in range(n)]

#Lightgrey background for Matplotlib/Seaborn charts since white hurts my eyes
plt.rcParams.update({
    "figure.facecolor": "lightblue",  #The area around the chart
    "axes.facecolor": "lightblue",    #The area inside the chart axes
    "savefig.facecolor": "lightblue"  #Ensures it stays grey if saved
})

#Data Loading
@st.cache_data
def load_learnsets():
    try:
        return pd.read_csv('pokemonLearnsets.csv', encoding='utf-8')
    except FileNotFoundError:
        return pd.DataFrame(columns=['pokemon_name','move_name','learn_method','level','version_group'])
 
learnsets_df = load_learnsets()

@st.cache_data
def load_smogon_sets():
    import json, os
    path = 'pokemon_smogon_sets.json'
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
 
smogon_sets = load_smogon_sets()
 
def _smogon_name(csv_name):
    name = str(csv_name).strip()
    if '\n' in name:
        name = name.split('\n')[1].strip()
    for prefix, suffix in [
        ('Alolan ',  '-Alola'), ('Galarian ', '-Galar'),
        ('Hisuian ', '-Hisui'), ('Paldean ',  '-Paldea'),
    ]:
        if name.startswith(prefix):
            return name[len(prefix):] + suffix
    for prefix in ('Mega ', 'Primal ', 'Gigantamax ', 'Eternamax '):
        if name.startswith(prefix):
            return None
    return name

def _move_tip_html(mv, mv_lookup, level=None):
    info   = mv_lookup.get(mv, {})
    mtype  = info.get('type', '--')
    dclass = info.get('damage_class', '--')
    pp     = info.get('pp', '--')
    pwr    = info.get('power', '--')
    acc    = info.get('accuracy', '--')
    effect = info.get('effect', '')
    level_str = f'<br>Learned at: Lv. {int(level)}' if level and pd.notna(level) and int(level) > 0 else ''
    tip_content = (
        f'<b>{mv}</b><br>'
        f'{mtype} &middot; {dclass}<br>'
        f'PP: {pp} &nbsp; Pwr: {pwr} &nbsp; Acc: {acc}'
        + level_str
        + (f'<br><i>{effect}</i>' if effect else '')
    )
    return (
        f"<span class='move-tip' style='display:inline;'>"
        f"<span style='color:#e0e0e0;'>{mv}</span>"
        f"<span class='tip-box'>{tip_content}</span>"
        f"</span>"
    )

@st.cache_data
def load_moves_meta():
    try:
        return pd.read_csv('pokemonMoves.csv', encoding='latin-1')
    except FileNotFoundError:
        return pd.DataFrame(columns=['name','accuracy','pp','power','priority',
                                     'type','generation','short_descripton','damage_class'])
 
moves_meta_df = load_moves_meta()
 
@st.cache_data
def build_moves_lookup(moves_meta):
    lookup = {}
    for _, row in moves_meta.iterrows():
        pwr = row.get('power', '')
        acc = row.get('accuracy', '')
        lookup[row['name']] = {
            'type':         str(row.get('type', '') or ''),
            'damage_class': str(row.get('damage_class', '') or ''),
            'pp':           int(row['pp']) if pd.notna(row.get('pp')) else '--',
            'power':        int(pwr) if pd.notna(pwr) and pwr != '' else '--',
            'accuracy':     int(acc) if pd.notna(acc) and acc != '' else '--',
            'effect':       str(row.get('short_descripton', '') or ''),
        }
    return lookup

@st.cache_data
def load_data():
    df = pd.read_csv('pokemon.csv', encoding='utf-8')
    
    #Pre-processing
    df['Type_1'] = df['Type'].apply(lambda x: str(x).split('\n')[0].strip() if pd.notna(x) else 'Unknown')
    df['Type_2'] = df['Type'].apply(lambda x: str(x).split('\n')[1].strip() if pd.notna(x) and '\n' in str(x) else None)
    return df

df = load_data()

if 'ml_stats' not in st.session_state:
    st.session_state.ml_stats = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']

# ============================================================================
# CACHED DATA PROCESSING FUNCTIONS
# ============================================================================

@st.cache_data
def get_top_pokemon_by_type(df, pokemon_type, n=10):
    """Get top N pokemon of a specific type by total stats"""
    return df[df['Type_1'] == pokemon_type].nlargest(n, 'Total')

@st.cache_data
def get_top_pokemon_by_stat(df, stat, n=10):
    """Get top N pokemon by a specific stat"""
    return df.nlargest(n, stat)

@st.cache_data
def get_type_distribution(df, column_name):
    """Calculate type distribution counts"""
    return df[column_name].value_counts().sort_values(ascending=False)

@st.cache_data
def get_average_stats_by_type(df):
    """Calculate average total stats by pokemon type"""
    return df.groupby('Type_1')['Total'].mean().sort_values(ascending=False)

@st.cache_data
def get_stat_correlation(df):
    """Calculate correlation matrix for pokemon stats"""
    return df[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']].corr()

@st.cache_data
def get_generation_avg_stats(df, base_only=False):
    """Return a list of (gen_num, region_name, avg_stats_dict) for all 9 generations, split by canonical National Dex ID ranges.

    base_only: if True, exclude alternate forms and Mega Evolutions.
    """
    GENERATIONS = [
        (1, "Kanto",  1,   151),
        (2, "Johto",  152, 251),
        (3, "Hoenn",  252, 386),
        (4, "Sinnoh", 387, 493),
        (5, "Unova",  494, 649),
        (6, "Kalos",  650, 721),
        (7, "Alola",  722, 809),
        (8, "Galar",  810, 905),
        (9, "Paldea", 906, 1025),
    ]
    STATS = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']

    #Keywords derived from every alternate form present in the dataset.
    #Organised by category so it's easy to extend if new forms are added.
    _FORM_KEYWORDS = [
        #Regional variants
        'Mega ', 'Primal ',
        'Alolan ', 'Galarian ', 'Hisuian ', 'Paldean ',

        #Common form suffixes
        ' Forme', ' Form', ' Mode', ' Style',
        ' Cloak', ' Breed', ' Mask', ' Rider',
        ' Rotom', ' Size', ' Plumage',

        #Fusions & special transformations
        'Eternamax', 'Ash-Greninja',
        'Black Kyurem', 'White Kyurem',
        'Hoopa Confined', 'Hoopa Unbound',
        'Dawn Wings Necrozma', 'Dusk Mane Necrozma', 'Ultra Necrozma',

        #Miscellaneous specific forms
        'Partner ',       #Partner Pikachu / Partner Eevee
        'Crowned ',       #Crowned Sword / Crowned Shield (Zacian/Zamazenta)
        'Own Tempo',      #Own Tempo Rockruff
        'Family of',      #Family of Three / Family of Four (Maushold)
        'Ice Face',       #Eiscue
        'Noice Face',     #Eiscue
        'Bloodmoon',      #Bloodmoon Ursaluna
        'Hero of Many Battles',  #Koraidon/Miraidon base form label

        #Gender variants
        # Meowstic, Indeedee, Basculegion, Oinkologne have a Female form.
        'Male', 'Female',
    ]

    if base_only:
        working_df = df[~df['Name'].apply(
            lambda n: (
                any(kw.lower() in str(n).lower() for kw in _FORM_KEYWORDS)
            )
        )]
    else:
        working_df = df

    results = []
    for gen_num, region, id_start, id_end in GENERATIONS:
        subset = working_df[(working_df['#'] >= id_start) & (working_df['#'] <= id_end)]
        if subset.empty:
            avg = {s: 0 for s in STATS}
        else:
            avg = {s: round(subset[s].mean(), 1) for s in STATS}
        results.append((gen_num, region, avg))
    return results

@st.cache_data
def get_pokemon_stats(df, name, categories):
    """Get stats for a specific pokemon"""
    return df[df['Name'] == name][categories].values.flatten().tolist()

@st.cache_data
def get_pokemon_moves(learnsets, name):
    """Return a dict {learn_method: [move_name, ...]} for a Pokemon, sorted alpha."""
    rows = learnsets[learnsets['pokemon_name'] == name]
    if rows.empty:
        return {}
    result = {}
    for method, grp in rows.groupby('learn_method'):
        result[method] = sorted(grp['move_name'].drop_duplicates().tolist())
    return result

@st.cache_data
def get_pokemon_abilities(df, name):
    """Return (normal_abilities_list, hidden_ability_str) for a Pokémon."""
    rows = df[df['Name'] == name]
    if rows.empty:
        return [], None
    row = rows.iloc[0]
    raw = str(row.get('Abilities', '') or '')
    normal = [a.strip() for a in raw.split(',') if a.strip() and a.strip() != 'N/A']
    hidden = str(row.get('Hidden_Ability', '') or '').strip()
    hidden = None if hidden in ('', 'None', 'N/A', 'nan') else hidden
    return normal, hidden

@st.cache_data
def perform_clustering(df, selected_features, k_val):
    """Perform K-Means clustering"""
    df_clustered = df.copy()

    # Compute per-Pokémon stat totals across only the selected features
    stat_totals = df_clustered[selected_features].sum(axis=1).replace(0, 1)

    # Build ratio columns: each stat as a fraction of the Pokémon's total
    ratio_cols = []
    for feat in selected_features:
        col = f'_ratio_{feat}'
        df_clustered[col] = df_clustered[feat] / stat_totals
        ratio_cols.append(col)

    # Scale ratios (they're already proportional but scaling helps KMeans treat each dimension equally regardless of natural variance)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_clustered[ratio_cols])

    # n_init=50 runs KMeans from 50 random seeds and picks the best result
    kmeans = KMeans(n_clusters=k_val, random_state=42, n_init=50)
    df_clustered['Cluster'] = kmeans.fit_predict(scaled)

    # Drop temporary ratio columns
    df_clustered.drop(columns=ratio_cols, inplace=True)

    return df_clustered

@st.cache_data
def get_cluster_summary(df_clustered, selected_features):
    """Calculate cluster summary statistics (raw stat averages per cluster)"""
    return df_clustered.groupby('Cluster')[selected_features].mean().round(0)

@st.cache_data
def label_archetypes(cluster_summary, selected_features):
    """Auto-label each cluster with a competitive archetype name based on which stats are most dominant relative to the cluster's own stat mix.
    """
    ICONS = {
        'Physical Sweeper': '⚔️',
        'Special Sweeper':  '✨',
        'Physical Wall':    '🛡️',
        'Special Wall':     '🔮',
        'Bulky Attacker':   '💪',
        'Bulky Special Attacker': '🔯',
        'Balanced':         '⚖️',
    }

    #Convert raw averages to per-Pokémon stat ratios at the cluster level
    normed = cluster_summary.div(cluster_summary.sum(axis=1), axis=0)

    #Deviation from the cross-cluster mean
    #Positive = this cluster emphasises this stat MORE than others do
    global_mean = normed.mean()
    dev = normed.sub(global_mean)

    def g(row, stat):
        return row.get(stat, 0)

    def score_dev(row, archetype):
        """Primary scoring — uses cross-cluster deviation so each cluster is
        evaluated relative to others rather than on absolute values.

        Archetype definitions:
          Physical Sweeper       — high Speed + Attack, low bulk
          Special Sweeper        — high Speed + Sp. Atk, low bulk
          Physical Wall          — high Defense (primary signal), low Speed + Sp. Atk
          Special Wall           — high HP + Sp. Def, low Speed + Attack
          Bulky Attacker         — high Attack + HP, low Speed + Sp. Atk
          Bulky Special Attacker — high Sp. Atk + HP, low Speed + Attack

        Note: Physical Wall uses Defense*2 so the defense-dominant cluster wins
        over the pure-HP cluster whose Defence deviation is low.
        """
        atk   = g(row, 'Attack')
        spatk = g(row, 'Sp. Atk')
        spd   = g(row, 'Speed')
        dfn   = g(row, 'Defense')
        spdef = g(row, 'Sp. Def')
        hp    = g(row, 'HP')

        if archetype == 'Physical Sweeper':
            return (spd + atk)     - (hp + dfn + spdef)
        if archetype == 'Special Sweeper':
            return (spd + spatk)   - (hp + dfn + spdef)
        if archetype == 'Physical Wall':
            return (dfn*2 + hp)    - (spd + spatk)
        if archetype == 'Special Wall':
            return (spdef*2 + hp)  - (spd + atk)
        if archetype == 'Bulky Attacker':
            return (atk + dfn)     - (spd + spatk)
        if archetype == 'Bulky Special Attacker':
            return (spatk + spdef) - (spd + atk)
        return 0
    
    def score_raw(row, archetype):
        """Fallback scoring for absorbed clusters — uses raw normalised stat ratios rather than deviation, which is more reliable when a cluster is
        defined by one extreme outlier stat (e.g. Blissey's HP=255 drowning out SpDef in deviation space)."""
        atk   = g(row, 'Attack')
        spatk = g(row, 'Sp. Atk')
        spd   = g(row, 'Speed')
        dfn   = g(row, 'Defense')
        spdef = g(row, 'Sp. Def')
        hp    = g(row, 'HP')

        if archetype == 'Physical Sweeper':
            return (spd + atk)     - (hp + dfn + spdef)
        if archetype == 'Special Sweeper':
            return (spd + spatk)   - (hp + dfn + spdef)
        if archetype == 'Physical Wall':
            return (dfn*2 + hp)    - (spd + spatk)
        if archetype == 'Special Wall':
            return (spdef*2 + hp)  - (spd + atk)
        if archetype == 'Bulky Attacker':
            return (atk + dfn)     - (spd + spatk)
        if archetype == 'Bulky Special Attacker':
            return (spatk + spdef) - (spd + atk)
        return 0

    archetypes   = list(ICONS.keys())
    cluster_ids  = list(dev.index)

    # Build score matrix using deviation-based scoring
    score_matrix = {
        cid: {arch: score_dev(dev.loc[cid], arch) for arch in archetypes}
        for cid in cluster_ids
    }

    # Phase 1 — greedy unique assignment for the 6 best-matching clusters.
    # k_val is set to 8 so KMeans forms tighter, more homogeneous groups,
    # giving the labelling better raw material to work with. The 2 extra
    # clusters beyond the 6 archetypes are absorbed in phase 2.
    labels             = {}
    remaining_clusters = set(cluster_ids)
    remaining_archetypes = set(archetypes)

    while remaining_archetypes:
        best_score = None
        best_cid   = None
        best_arch  = None

        for cid in remaining_clusters:
            for arch in remaining_archetypes:
                s = score_matrix[cid][arch]
                if best_score is None or s > best_score:
                    best_score = s
                    best_cid   = cid
                    best_arch  = arch

        labels[best_cid] = f"{ICONS[best_arch]} {best_arch}"
        remaining_clusters.discard(best_cid)
        remaining_archetypes.discard(best_arch)

    # Phase 2 — absorbed clusters get the archetype label whose raw-ratio
    # score is highest. Raw ratios are used here because extreme HP outliers
    # (e.g. Blissey, Wobbuffet) distort the deviation values for their cluster.
    for cid in remaining_clusters:
        raw_row   = normed.loc[cid]
        best_arch = max(archetypes, key=lambda a: score_raw(raw_row, a))
        labels[cid] = f"{ICONS[best_arch]} {best_arch}"

    return labels

@st.cache_data
def add_archetype_axes(df_clustered, selected_features):
    """ Derive two axes that map directly to competitive roles

      X  =  Attack - Sp. Atk
             left  -> pure Special Attacker
             right -> pure Physical Attacker

      Y  =  Speed - mean(Defense, Sp. Def, HP)
             bottom -> bulky Wall
             top    -> fast Sweeper

    Falls back to zero for any stat not in selected_features.
    """
    out = df_clustered.copy()

    def g(col):
        return out[col] if col in selected_features else pd.Series(0, index=out.index)

    out['_axis_x'] = g('Attack') - g('Sp. Atk')

    bulk_cols = [s for s in ['Defense', 'Sp. Def', 'HP'] if s in selected_features]
    avg_bulk = sum(g(s) for s in bulk_cols) / max(len(bulk_cols), 1)
    out['_axis_y'] = g('Speed') - avg_bulk

    return out

@st.cache_data
def perform_dbscan(df, selected_features, eps=2, min_samples=8):
    """Two-phase outlier-aware clustering:
    Phase 1 — DBSCAN on raw scaled stats.
      Raw values (not ratios) are used so that Pokemon with extreme absolute stats (Blissey HP=255, Shuckle Def=230, Eternatus totals) sit far from the main cloud and are correctly flagged as noise (label = -1).
    Phase 2 — K-Means (k=6) on the clean non-outlier subset.
      Ratio features are used here so archetypes reflect stat *shape* rather than overall power level — a weak Rapidash and a strong Arcanine should both land in Physical Sweeper even though their raw stats differ.

    Returns
    df_result: DataFrame with columns 'Is_Outlier' (bool) and 'Cluster' (int, -1 for outliers)
    core_summary: cluster mean stats (non-outliers only)
    """
    df_result = df.copy()

    # --- Phase 1: DBSCAN outlier detection ---
    raw_scaled = StandardScaler().fit_transform(df_result[selected_features])
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(raw_scaled)
    df_result['Is_Outlier'] = db.labels_ == -1
    df_result['Cluster']    = db.labels_   #-1 = outlier, 0+ = DBSCAN cluster (discarded after phase 1)

    # --- Phase 2: K-Means on non-outliers using ratio features ---
    core_mask = ~df_result['Is_Outlier']
    core_df   = df_result[core_mask].copy()

    stat_totals = core_df[selected_features].sum(axis=1).replace(0, 1)
    ratio_cols  = []
    for feat in selected_features:
        col = f'_ratio_{feat}'
        core_df[col] = core_df[feat] / stat_totals
        ratio_cols.append(col)

    core_scaled = StandardScaler().fit_transform(core_df[ratio_cols])
    km = KMeans(n_clusters=6, random_state=42, n_init=50)
    core_df['Cluster'] = km.fit_predict(core_scaled)
    core_df.drop(columns=ratio_cols, inplace=True)

    # Write cluster assignments back into the full DataFrame
    df_result.loc[core_mask, 'Cluster'] = core_df['Cluster']

    core_summary = core_df.groupby('Cluster')[selected_features].mean().round(0)
    return df_result, core_summary


@st.cache_data
def label_archetypes_dbscan(core_summary, selected_features):
    """Same greedy unique-assignment labelling as K-Means, tuned for
    the cleaner cluster shapes that come from running on outlier-free data.
    With outliers removed k=6 maps naturally onto 6 archetypes so no
    absorption phase is needed."""
    ICONS = {
        'Physical Sweeper':       '⚔️',
        'Special Sweeper':        '✨',
        'Physical Wall':          '🛡️',
        'Special Wall':           '🔮',
        'Bulky Attacker':         '💪',
        'Bulky Special Attacker': '🔯',
    }

    normed      = core_summary.div(core_summary.sum(axis=1), axis=0)
    global_mean = normed.mean()
    dev         = normed.sub(global_mean)

    def g(row, stat):
        return row.get(stat, 0)

    def score(row, archetype):
        atk   = g(row, 'Attack');  spatk = g(row, 'Sp. Atk')
        spd   = g(row, 'Speed');   dfn   = g(row, 'Defense')
        spdef = g(row, 'Sp. Def'); hp    = g(row, 'HP')
        if archetype == 'Physical Sweeper':       return (spd + atk)     - (hp + dfn + spdef)
        if archetype == 'Special Sweeper':        return (spd + spatk)   - (hp + dfn + spdef)
        if archetype == 'Physical Wall':          return (dfn*2 + hp)    - (spd + spatk)
        if archetype == 'Special Wall':           return (spdef*2 + hp)  - (spd + atk)
        if archetype == 'Bulky Attacker':         return (atk + dfn)     - (spd + spatk)
        if archetype == 'Bulky Special Attacker': return (spatk + spdef) - (spd + atk)
        return 0

    archetypes  = list(ICONS.keys())
    cluster_ids = list(dev.index)
    score_matrix = {cid: {arch: score(dev.loc[cid], arch) for arch in archetypes} for cid in cluster_ids}

    labels = {}
    remaining_c = set(cluster_ids)
    remaining_a = set(archetypes)
    while remaining_a:
        best_score, best_cid, best_arch = None, None, None
        for cid in remaining_c:
            for arch in remaining_a:
                s = score_matrix[cid][arch]
                if best_score is None or s > best_score:
                    best_score, best_cid, best_arch = s, cid, arch
        labels[best_cid] = f"{ICONS[best_arch]} {best_arch}"
        remaining_c.discard(best_cid)
        remaining_a.discard(best_arch)

    return labels

@st.cache_data
def get_sprite_path(pokemon_name, df):
    try:
        sprite_path = df[df['Name'] == pokemon_name]['Local_Sprite'].iloc[0]     
        return sprite_path if pd.notna(sprite_path) and sprite_path != '' else None
    except:
        return None
    
@st.cache_data
def get_type_sprite_paths(pokemon_name, df):
    try:
        row = df[df['Name'] == pokemon_name].iloc[0]
        types = []
        for col in ['Type_1', 'Type_2']:
            val = row.get(col)
            if pd.notna(val) and str(val).strip() not in ('', 'None'):
                type_name = str(val).strip().lower()
                types.append((str(val).strip(), f"typeSprites/{type_name}.png"))
        return types
    except:
        return []
    
@st.cache_data
def get_ability_counts(df, type_filter=None):
    """Count how many Pokemon can have each ability (normal + hidden combined).
    Returns a DataFrame with columns: Ability, Count, Category.
    Category is 'Normal Only', 'Hidden Only', or 'Normal & Hidden'"""
    from collections import Counter
    if type_filter and type_filter != "All Types":
        working = df[df['Type_1'] == type_filter]
    else:
        working = df
 
    normal_counts = Counter()
    hidden_counts = Counter()
    for _, row in working.iterrows():
        raw = str(row.get('Abilities', '') or '')
        for a in raw.split(','):
            a = a.strip()
            if a and a not in ('N/A', 'nan', ''):
                normal_counts[a] += 1
        hid = str(row.get('Hidden_Ability', '') or '').strip()
        if hid and hid not in ('N/A', 'nan', 'None', ''):
            hidden_counts[hid] += 1
 
    all_abs = sorted(set(normal_counts) | set(hidden_counts))
    rows = []
    for a in all_abs:
        n, h = normal_counts.get(a, 0), hidden_counts.get(a, 0)
        total = n + h
        if n > 0 and h > 0:   cat = 'Normal & Hidden'
        elif h > 0:            cat = 'Hidden Only'
        else:                  cat = 'Normal Only'
        rows.append({'Ability': a, 'Count': total, 'Category': cat})
 
    return pd.DataFrame(rows).sort_values('Count', ascending=False).reset_index(drop=True)

@st.cache_data
def get_move_counts(learnsets, type_filter=None, pokemon_df=None):
    """Count how many Pokemon can learn each move, grouped by primary learn method.
    Returns a DataFrame with columns: Move, Count, Category."""
    working = learnsets.copy()
    if type_filter and type_filter != "All Types" and pokemon_df is not None:
        names_of_type = pokemon_df[pokemon_df['Type_1'] == type_filter]['Name'].tolist()
        working = working[working['pokemon_name'].isin(names_of_type)]
 
    #Primary category per move = the learn method with the most Pokemon for that move
    method_order = {'level-up': 0, 'machine': 1, 'egg': 2, 'tutor': 3, 'form-change': 4}
 
    rows = []
    for move, grp in working.groupby('move_name'):
        count = grp['pokemon_name'].nunique()
        #Assign category by the most "natural" method present
        methods_present = set(grp['learn_method'].unique())
        for m in ['level-up', 'machine', 'tutor', 'egg', 'form-change']:
            if m in methods_present:
                category = m.replace('-', ' ').title()
                break
        else:
            category = 'Other'
        rows.append({'Move': move, 'Count': count, 'Category': category})
 
    return pd.DataFrame(rows).sort_values('Count', ascending=False).reset_index(drop=True)
    
@st.cache_data
def get_stat_distribution_by_type(df, stat):
    """Return a dict {type: sorted_values_array} for box plot, sorted by median desc"""
    groups = {}
    for t, grp in df.groupby('Type_1'):
        vals = grp[stat].dropna().values
        if len(vals) > 0:
            groups[t] = vals
    return dict(sorted(groups.items(), key=lambda x: float(np.median(x[1])), reverse=True))

@st.cache_data
def get_type_composition_by_region(df):
    """Return a DataFrame (index=region, columns=type, values=% of that gen's pokemon)"""
    GENERATIONS = [
        (1, "Kanto",  1,   151),
        (2, "Johto",  152, 251),
        (3, "Hoenn",  252, 386),
        (4, "Sinnoh", 387, 493),
        (5, "Unova",  494, 649),
        (6, "Kalos",  650, 721),
        (7, "Alola",  722, 809),
        (8, "Galar",  810, 905),
        (9, "Paldea", 906, 1025),
    ]
    all_types = sorted(df['Type_1'].dropna().unique())
    rows = {}
    for gen_num, region, id_start, id_end in GENERATIONS:
        subset = df[(df['#'] >= id_start) & (df['#'] <= id_end)]
        total  = len(subset)
        if total == 0:
            rows[f"Gen {gen_num}\n{region}"] = {t: 0.0 for t in all_types}
        else:
            counts = subset['Type_1'].value_counts()
            rows[f"Gen {gen_num}\n{region}"] = {
                t: round(counts.get(t, 0) / total * 100, 1) for t in all_types
            }
    return pd.DataFrame(rows).T   #regions as rows, types as columns


@st.cache_data
def get_top_by_physical(df, column, ascending=False, n=10):
    """Top N Pokémon by Height or Weight, dropping rows with no data.
    Coercion happens here (not just in load_data) so it runs even when the
    Streamlit data cache was built before the coercion lines were added."""
    working = df.copy()
    working[column] = pd.to_numeric(working[column], errors='coerce')
    valid = working[working[column].notna()]
    return valid.nsmallest(n, column) if ascending else valid.nlargest(n, column)

@st.cache_data
def get_top_legendary(df, n=10):
    """Top N Legendary Pokémon by Total stats."""
    return df[df['Is_Legendary'] == 1].nlargest(n, 'Total')

@st.cache_data
def get_top_mythical(df, n=10):
    """Top N Mythical Pokémon by Total stats."""
    return df[df['Is_Mythical'] == 1].nlargest(n, 'Total')
# ============================================================================
# UI STARTS HERE
# ============================================================================

#Segmented Control Override
st.markdown("""
<style>
/* Unselected: light sky blue, dark text */
button[data-testid="stBaseButton-segmented_control"] {
    background-color: #5bb8f5 !important;
    color: #111111 !important;
    border: 2px solid #2980b9 !important;
    font-weight: 600 !important;
    transition: background-color 0.15s ease, border-color 0.15s ease !important;
}
/*Hover */
button[data-testid="stBaseButton-segmented_control"]:hover {
    background-color: #3fa3e8 !important;
    border-color: #1a6fa0 !important;
}

/* Selected: deep Pokédex navy, white text */
button[data-testid="stBaseButton-segmented_controlActive"] {
    background-color: #1565c0 !important;
    color: #ffffff !important;
    border: 2px solid #0d3f7a !important;
    font-weight: 600 !important;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.35) !important;
}

button[data-testid="stBaseButton-segmented_controlActive"]:hover {
    background-color: #1158a8 !important;
}
            
/*Light-grey accent for the popover element */
/* Trigger button */
button[data-testid="stPopoverButton"] {
    background-color: #d6d6d6 !important;
    color: #111111 !important;
    border: 2px solid #a0a0a0 !important;
    font-weight: 600 !important;
    width: fit-content !important;
    transition: background-color 0.15s ease !important;
}
button[data-testid="stPopoverButton"]:hover {
    background-color: #bebebe !important;
    border-color: #808080 !important;
}
            
div[data-testid="stPopover"] {
    width: fit-content !important;
    display: inline-flex !important;
}       

/* Floating panel background */
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div {
    background-color: #ececec !important;
    border: 2px solid #a0a0a0 !important;
    border-radius: 6px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25) !important;       
}

/* Radio labels inside the popover */
div[data-testid="stRadio"] label p {
    color: #111111 !important;
    font-weight: 500 !important;
}
            
div[data-testid="stRadio"] [data-testid="stWidgetLabel"] p {
    color: #111111 !important;
    opacity: 0.8; /* Makes the header slightly distinct */
}
            
.move-tip {
    position: relative;
    cursor: default;
    display: inline-block;
}
.move-tip .tip-box {
    visibility: hidden;
    opacity: 0;
    background-color: #1e1e1e;
    color: #e0e0e0;
    font-family: monospace;
    font-size: 0.72rem;
    line-height: 1.6;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 8px 10px;
    position: absolute;
    z-index: 9999;
    left: 50%;
    transform: translateX(-50%);
    bottom: calc(100% + 6px);
    min-width: 200px;
    max-width: 280px;
    white-space: normal;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    transition: opacity 0.15s ease;
    pointer-events: none;
}
.move-tip:hover .tip-box {
    visibility: visible;
    opacity: 1;
}
</style>
""", unsafe_allow_html=True)

# Title Section
st.markdown("""
<div style="
    background-color: #242424;
    border: 12px solid #d0d0d0;
    border-radius: 6px;
    box-shadow: 0 0 0 3px #c1c1c1, inset 0 2px 6px rgba(0,0,0,0.35);
    padding: 12px 32px 14px 32px;
    margin-bottom: 24px;
    width: 100%;
    box-sizing: border-box;
">
    <p style="
        font-family: monospace;
        font-size: 2.4rem;
        font-weight: 900;
        color: #ffffff;
        margin: 0 0 2px 0;
        letter-spacing: 2px;
        text-shadow: 1px 1px 0px #ffffff;
    ">The Poké-DAT</p>
    <p style="
        font-family: monospace;
        font-size: 0.95rem;
        color: #ffffff;
        margin: 0;
        letter-spacing: 1px;
    ">▶ A Data Analysis &amp; Visualization Tool</p>
</div>
""", unsafe_allow_html=True)

# Create categorized main tabs
main_tabs = st.tabs([
    "Rankings", 
    "Trends", 
    "Relationships",
    "Machine Learning",
    "Pokédex Lookup"
])

#MAIN TAB 1: RANKINGS
with main_tabs[0]:
    mode = st.segmented_control ("View Rankings By:", ["Type", "Specific Stat", "Height", "Weight", "Legendary", "Mythical"], default="Type", key="mode" )
    
    if mode == "Type":
        #SUB-SECTION: Top 10 by Type 
        st.header("Top 10 Most Powerful Pokémon by Type")
        unique_types = sorted(df['Type_1'].unique())
        with st.popover(f"Filter by Type: {st.session_state.get('type_choice', 'BUG')}"):
            selected_type = st.radio(
                "Select a Pokémon Type:", 
                unique_types, 
                key="type_choice"
            )

        #Get filtered data
        filtered_df = get_top_pokemon_by_type(df, selected_type, 10)
        
        #Chart at full width
        num_items = len(filtered_df)
        fig_height = num_items * 0.4
        
        left_pad, chart_area, right_pad = st.columns([1, 8, 1])
        with chart_area:
            fig, ax = plt.subplots(figsize=(8, fig_height))
            colors = get_pokedex_colors(num_items)
            
            y_positions = range(num_items)
            ax.barh(y_positions, filtered_df['Total'], color=colors)
            
            for i, v in enumerate(filtered_df['Total']):
                ax.text(v + 0.5, i, f'{int(v)}', va='center', 
                       fontweight='bold', fontsize=8)
            
            ax.set_yticks(y_positions)
            ax.set_yticklabels([f"#{i+1}" for i in range(num_items)], 
                              fontsize=11, fontweight='bold')
            ax.invert_yaxis()
            ax.set_xlabel('Total Stats', fontweight='bold', fontsize=11)
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        #Sprites in a horizontal strip below the chart — one column per Pokémon
        st.markdown("### Pokémon")
        sprite_cols = st.columns(num_items)
        for rank, ((idx, row), col) in enumerate(zip(filtered_df.iterrows(), sprite_cols), 1):
            with col:
                sprite_path = get_sprite_path(row['Name'], df)
                st.caption(f"**#{rank}**")
                if sprite_path:
                    st.image(sprite_path, use_container_width=True)
                else:
                    st.markdown("<div style='text-align:center; font-size:24px;'>🎮</div>", unsafe_allow_html=True)
                st.caption(f"{row['Name'].replace(chr(10), ' ')}")
            
            
    elif mode == "Specific Stat":
        #SUB-SECTION: Specific Stats
        st.header("Top 10 Pokémon by Specific Stat")
        stat_choice = st.segmented_control(
            "Select Stat", 
            ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed'],
            default='Total'
        )

        top_stat_df = get_top_pokemon_by_stat(df, stat_choice, 10)
        
        #Chart at full width
        num_items = len(top_stat_df)
        fig_height = num_items * 0.4

        left_pad, chart_area, right_pad = st.columns([1, 8, 1])
        with chart_area:
            fig2, ax2 = plt.subplots(figsize=(8, fig_height))
            colors2 = get_pokedex_colors(num_items)
            
            y_positions = range(num_items)
            ax2.barh(y_positions, top_stat_df[stat_choice], color=colors2)
            
            for i, v in enumerate(top_stat_df[stat_choice]):
                ax2.text(v + 0.5, i, f'{int(v)}', va='center', 
                        fontweight='bold', fontsize=8)
            
            ax2.set_yticks(y_positions)
            ax2.set_yticklabels([f"#{i+1}" for i in range(num_items)], 
                               fontsize=11, fontweight='bold')
            ax2.invert_yaxis()
            ax2.set_xlabel(f'{stat_choice} Stats', fontweight='bold', fontsize=11)
            ax2.grid(axis='x', alpha=0.3, linestyle='--')
            
            plt.tight_layout()
            st.pyplot(fig2)

        #Sprites in a horizontal strip below the chart — one column per Pokémon
        st.markdown("### Pokémon")
        sprite_cols = st.columns(num_items)
        for rank, ((idx, row), col) in enumerate(zip(top_stat_df.iterrows(), sprite_cols), 1):
            with col:
                sprite_path = get_sprite_path(row['Name'], df)
                st.caption(f"**#{rank}**")
                if sprite_path:
                    st.image(sprite_path, use_container_width=True)
                else:
                    st.markdown("<div style='text-align:center; font-size:24px;'>🎮</div>", unsafe_allow_html=True)
                st.caption(f"{row['Name'].replace(chr(10), ' ')}")


    elif mode == "Height":
        #SUB-SECTION: Top 10 by Height
        st.header("Top 10 Pokémon by Height")

        height_dir = st.segmented_control ("Height ranking:", ["Tallest", "Shortest"], default="Tallest", key="height_dir" )

        st.divider()

        #Height chart
        st.subheader(f"{'Top 10 Tallest' if height_dir == 'Tallest' else 'Top 10 Shortest'} Pokémon")
        h_df = get_top_by_physical(df, 'Height_m', ascending=(height_dir == "Shortest"))
        h_num = len(h_df)

        left_pad, chart_area, right_pad = st.columns([1, 8, 1])
        with chart_area:
            fig_h, ax_h = plt.subplots(figsize=(8, h_num * 0.4))
            colors_h = get_pokedex_colors(h_num)
            ax_h.barh(range(h_num), h_df['Height_m'], color=colors_h)
            _h_max = h_df['Height_m'].max()
            _h_pad = _h_max * 0.30
            ax_h.set_xlim(0, _h_max + _h_pad)
            ax_h.xaxis.set_major_locator(plt.MaxNLocator(nbins=6, steps=[1,2,5,10]))
            ax_h.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:g}'))
            for i, v in enumerate(h_df['Height_m']):
                ax_h.text(v + _h_max * 0.02, i, f'{v:g} m', va='center', fontweight='bold', fontsize=8)
            ax_h.set_yticks(range(h_num))
            ax_h.set_yticklabels([f"#{i+1}" for i in range(h_num)], fontsize=11, fontweight='bold')
            ax_h.invert_yaxis()
            ax_h.set_xlabel('Height (m)', fontweight='bold', fontsize=11)
            ax_h.grid(axis='x', alpha=0.3, linestyle='--')
            plt.tight_layout()
            st.pyplot(fig_h)

        st.markdown("### Pokémon")
        sprite_cols_h = st.columns(h_num)
        for rank, ((idx, row), col) in enumerate(zip(h_df.iterrows(), sprite_cols_h), 1):
            with col:
                sprite_path = get_sprite_path(row['Name'], df)
                st.caption(f"**#{rank}**")
                if sprite_path:
                    st.image(sprite_path, use_container_width=True)
                else:
                    st.markdown("<div style='text-align:center;font-size:24px;'>Unavailable</div>", unsafe_allow_html=True)
                st.caption(f"{row['Name'].replace(chr(10), ' ')}")

    elif mode == "Weight":
        #SUB-SECTION: Top 10 by Weight
        st.header("Top 10 Pokémon by Weight")
        weight_dir = st.segmented_control ("Weight ranking:", ["Heaviest", "Lightest"], default="Heaviest", key="weight_dir" )
        st.divider()
         #Weight chart
        st.subheader(f"{'Top 10 Heaviest' if weight_dir == 'Heaviest' else 'Top 10 Lightest'} Pokémon")
        w_df = get_top_by_physical(df, 'Weight_kg', ascending=(weight_dir == "Lightest"))
        w_num = len(w_df)

        left_pad2, chart_area2, right_pad2 = st.columns([1, 8, 1])
        with chart_area2:
            fig_w, ax_w = plt.subplots(figsize=(8, w_num * 0.4))
            colors_w = get_pokedex_colors(w_num)
            ax_w.barh(range(w_num), w_df['Weight_kg'], color=colors_w)
            _w_max = w_df['Weight_kg'].max()
            _w_pad = _w_max * 0.30
            ax_w.set_xlim(0, _w_max + _w_pad)
            ax_w.xaxis.set_major_locator(plt.MaxNLocator(nbins=6, steps=[1,2,5,10]))
            ax_w.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:g}'))
            for i, v in enumerate(w_df['Weight_kg']):
                ax_w.text(v + _w_max * 0.02, i, f'{v:g} kg', va='center', fontweight='bold', fontsize=8)
            ax_w.set_yticks(range(w_num))
            ax_w.set_yticklabels([f"#{i+1}" for i in range(w_num)], fontsize=11, fontweight='bold')
            ax_w.invert_yaxis()
            ax_w.set_xlabel('Weight (kg)', fontweight='bold', fontsize=11)
            ax_w.grid(axis='x', alpha=0.3, linestyle='--')
            plt.tight_layout()
            st.pyplot(fig_w)

        st.markdown("### Pokémon")
        sprite_cols_w = st.columns(w_num)
        for rank, ((idx, row), col) in enumerate(zip(w_df.iterrows(), sprite_cols_w), 1):
            with col:
                sprite_path = get_sprite_path(row['Name'], df)
                st.caption(f"**#{rank}**")
                if sprite_path:
                    st.image(sprite_path, use_container_width=True)
                else:
                    st.markdown("<div style='text-align:center;font-size:24px;'>Unavailable</div>", unsafe_allow_html=True)
                st.caption(f"{row['Name'].replace(chr(10), ' ')}")       

    elif mode == "Legendary":
        #SUB-SECTION: Top 10 Legendary
        st.header("Top 10 Legendary Pokémon by Total Stats")

        leg_df  = get_top_legendary(df)
        leg_num = len(leg_df)

        left_pad, chart_area, right_pad = st.columns([1, 8, 1])
        with chart_area:
            fig_l, ax_l = plt.subplots(figsize=(8, leg_num * 0.4))
            colors_l = get_pokedex_colors(leg_num)
            ax_l.barh(range(leg_num), leg_df['Total'], color=colors_l)
            for i, v in enumerate(leg_df['Total']):
                ax_l.text(v + 0.5, i, f'{int(v)}', va='center', fontweight='bold', fontsize=8)
            ax_l.set_yticks(range(leg_num))
            ax_l.set_yticklabels([f"#{i+1}" for i in range(leg_num)], fontsize=11, fontweight='bold')
            ax_l.invert_yaxis()
            ax_l.set_xlabel('Total Stats', fontweight='bold', fontsize=11)
            ax_l.grid(axis='x', alpha=0.3, linestyle='--')
            plt.tight_layout()
            st.pyplot(fig_l)

        st.markdown("### Pokémon")
        sprite_cols_l = st.columns(leg_num)
        for rank, ((idx, row), col) in enumerate(zip(leg_df.iterrows(), sprite_cols_l), 1):
            with col:
                sprite_path = get_sprite_path(row['Name'], df)
                st.caption(f"**#{rank}**")
                if sprite_path:
                    st.image(sprite_path, use_container_width=True)
                else:
                    st.markdown("<div style='text-align:center;font-size:24px;'>Unavailable</div>", unsafe_allow_html=True)
                st.caption(f"{row['Name'].replace(chr(10), ' ')}")

    elif mode == "Mythical":

        #SUB-SECTION: Top 10 Mythical
        st.header("Top 10 Mythical Pokémon by Total Stats")
        myth_df  = get_top_mythical(df)
        myth_num = len(myth_df)

        left_pad2, chart_area2, right_pad2 = st.columns([1, 8, 1])
        with chart_area2:
            fig_m, ax_m = plt.subplots(figsize=(8, myth_num * 0.4))
            colors_m = get_pokedex_colors(myth_num)
            ax_m.barh(range(myth_num), myth_df['Total'], color=colors_m)
            for i, v in enumerate(myth_df['Total']):
                ax_m.text(v + 0.5, i, f'{int(v)}', va='center', fontweight='bold', fontsize=8)
            ax_m.set_yticks(range(myth_num))
            ax_m.set_yticklabels([f"#{i+1}" for i in range(myth_num)], fontsize=11, fontweight='bold')
            ax_m.invert_yaxis()
            ax_m.set_xlabel('Total Stats', fontweight='bold', fontsize=11)
            ax_m.grid(axis='x', alpha=0.3, linestyle='--')
            plt.tight_layout()
            st.pyplot(fig_m)

        st.markdown("### Pokémon")
        sprite_cols_m = st.columns(myth_num)
        for rank, ((idx, row), col) in enumerate(zip(myth_df.iterrows(), sprite_cols_m), 1):
            with col:
                sprite_path = get_sprite_path(row['Name'], df)
                st.caption(f"**#{rank}**")
                if sprite_path:
                    st.image(sprite_path, use_container_width=True)
                else:
                    st.markdown("<div style='text-align:center;font-size:24px;'>Unavailable</div>", unsafe_allow_html=True)
                st.caption(f"{row['Name'].replace(chr(10), ' ')}")


#MAIN TAB 2: TRENDS
with main_tabs[1]:
    trend_mode = st.segmented_control ("View Trends By:", ["Type Distribution", "Average Power Level", "Ability Distribution", "Move Distribution", "Stat Distribution by Type", "Type Composition by Region", "Base Stat Averages by Generation"], default="Type Distribution", key="trend_mode" )
    
    if trend_mode == "Type Distribution":
        #SUB-SECTION: Distribution
        st.header("Pokémon Type Distribution")
        dist_choice = st.segmented_control("Show Distribution for:", ["Primary Typing (Type 1)", "Secondary Typing (Type 2)"], default="Primary Typing (Type 1)", key="dist_choice")

        left, mid, right = st.columns([1, 5, 1])
        with mid:
            col_name = 'Type_1' if dist_choice == "Primary Typing (Type 1)" else 'Type_2'
            
            type_counts = get_type_distribution(df, col_name)
            
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            type_counts.plot(kind='bar', color='#EF5350', ax=ax3)

            ax3.set_title(f"Distribution of {dist_choice}", fontsize=10, fontweight='bold')
            ax3.set_xlabel(dist_choice, fontsize=8)
            ax3.set_ylabel("Count", fontsize=8)

            for i, v in enumerate(type_counts.values):
                ax3.text(i, v, str(v), ha='center', va='bottom', fontweight='bold', fontsize=8)
            st.pyplot(fig3)

    elif trend_mode == "Ability Distribution":
        st.header("Ability Distribution")
        st.write(
            "A treemap of every ability in the game, sized by how many Pokemon can have it. "
            "Colour shows whether it appears as a normal ability, hidden ability, or both."
        )
 
        #Type filter
        _ab_types = ["All Types"] + sorted(df['Type_1'].dropna().unique().tolist())
        ab_type_filter = st.selectbox(
            "Filter by Primary Type", _ab_types,
            key="ab_type_filter",
            index=0
        )
 
        ab_df = get_ability_counts(df, ab_type_filter)
 
        #Colour map: Normal Only = Pokedex blue, Hidden Only = purple, Both = teal
        _cat_colors = {
            'Normal Only':    '#42A5F5',
            'Hidden Only':    '#AB47BC',
            'Normal & Hidden':'#26A69A',
        }
        ab_df['Color'] = ab_df['Category'].map(_cat_colors)
 
        fig_tree = px.treemap(
            ab_df,
            path=['Category', 'Ability'],
            values='Count',
            color='Category',
            color_discrete_map=_cat_colors,
            custom_data=['Ability', 'Count', 'Category'],
        )
        fig_tree.update_traces(
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Pokemon with this ability: %{customdata[1]}<br>"
                "Type: %{customdata[2]}<extra></extra>"
            ),
            textinfo='label+value',
            textfont=dict(size=12),
        )
        fig_tree.update_layout(
            height=650,
            margin=dict(l=10, r=10, t=40, b=10),
            font=dict(color='black'),
        )
        st.plotly_chart(fig_tree, use_container_width=True)
 
        #Summary stats beneath the treemap
        n_abilities = len(ab_df)
        top3 = ab_df.head(3)
        only_one = (ab_df['Count'] == 1).sum()
 
        def _stat_card(label, value, subtitle=None):
            sub_html = (
                f"<p style='font-family:monospace; font-size:0.8rem; "
                f"color:#aaaaaa; margin:4px 0 0 0; letter-spacing:1px;'>"
                f"{subtitle}</p>"
            ) if subtitle else ""
            st.markdown(f"""
<div style="
    background-color: #242424;
    border: 8px solid #d0d0d0;
    border-radius: 6px;
    box-shadow: 0 0 0 2px #c1c1c1, inset 0 2px 6px rgba(0,0,0,0.35);
    padding: 14px 20px 16px 20px;
    text-align: center;
">
    <p style="
        font-family: monospace;
        font-size: 0.75rem;
        color: #aaaaaa;
        margin: 0 0 6px 0;
        letter-spacing: 2px;
        text-transform: uppercase;
    ">{label}</p>
    <p style="
        font-family: monospace;
        font-size: 1.6rem;
        font-weight: 900;
        color: #ffffff;
        margin: 0;
        letter-spacing: 1px;
        text-shadow: 1px 1px 0px #ffffff55;
    ">{value}</p>
    {sub_html}
</div>
""", unsafe_allow_html=True)
 
        sum_col1, sum_col2, sum_col3 = st.columns(3)
        with sum_col1:
            _stat_card("▶ Unique Abilities",  n_abilities)
        with sum_col2:
            _stat_card("▶ Most Common", top3.iloc[0]['Ability'], f"{top3.iloc[0]['Count']} Pokémon")
        with sum_col3:
            _stat_card("▶ Signature Abilities", only_one, "held by exactly 1 Pokémon")

    elif trend_mode == "Move Distribution":
        st.header("Move Distribution")
        st.write(
            "A treemap of every move in the game, sized by how many Pokemon can learn it. "
            "Colour shows the primary way each move is learned."
        )
 
        _mv_types = ["All Types"] + sorted(df['Type_1'].dropna().unique().tolist())
        mv_type_filter = st.selectbox(
            "Filter by Primary Type", _mv_types,
            key="mv_type_filter",
            index=0
        )
 
        mv_df = get_move_counts(learnsets_df, mv_type_filter, df)
 
        _mv_colors = {
            'Level Up':    '#EF5350',
            'Machine':     '#42A5F5',
            'Tutor':       '#26A69A',
            'Egg':         '#AB47BC',
            'Form Change': '#FFA726',
        }
 
        fig_mv = px.treemap(
            mv_df,
            path=['Category', 'Move'],
            values='Count',
            color='Category',
            color_discrete_map=_mv_colors,
            custom_data=['Move', 'Count', 'Category'],
        )
        fig_mv.update_traces(
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Pokemon that can learn this: %{customdata[1]}<br>"
                "Primary method: %{customdata[2]}<extra></extra>"
            ),
            textinfo='label+value',
            textfont=dict(size=12),
        )
        fig_mv.update_layout(
            height=650,
            margin=dict(l=10, r=10, t=40, b=10),
            font=dict(color='black'),
        )
        st.plotly_chart(fig_mv, use_container_width=True)
 
        # Summary stat cards
        n_moves     = len(mv_df)
        top_move    = mv_df.iloc[0]
        signature   = (mv_df['Count'] == 1).sum()
 
        def _mv_stat_card(label, value, subtitle=None):
            sub_html = (
                f"<p style='font-family:monospace; font-size:0.8rem; "
                f"color:#aaaaaa; margin:4px 0 0 0; letter-spacing:1px;'>"
                f"{subtitle}</p>"
            ) if subtitle else ""
            st.markdown(f"""
<div style="
    background-color: #242424;
    border: 8px solid #d0d0d0;
    border-radius: 6px;
    box-shadow: 0 0 0 2px #c1c1c1, inset 0 2px 6px rgba(0,0,0,0.35);
    padding: 14px 20px 16px 20px;
    text-align: center;
">
    <p style="font-family:monospace; font-size:0.75rem; color:#aaaaaa;
              margin:0 0 6px 0; letter-spacing:2px; text-transform:uppercase;">{label}</p>
    <p style="font-family:monospace; font-size:1.6rem; font-weight:900;
              color:#ffffff; margin:0; letter-spacing:1px;
              text-shadow:1px 1px 0px #ffffff55;">{value}</p>
    {sub_html}
</div>
""", unsafe_allow_html=True)
 
        c1, c2, c3 = st.columns(3)
        with c1:
            _mv_stat_card("Unique Moves", n_moves, " ")
        with c2:
            _mv_stat_card("Most Widespread", top_move['Move'], f"{top_move['Count']} Pokémon")
        with c3:
            _mv_stat_card("Signature Moves", signature, "learned by exactly 1 Pokémon")


    elif trend_mode == "Average Power Level":
        #SUB-SECTION: Average Power Level
        st.header("Average Power Level by Pokémon Type")
        left, mid, right = st.columns([2, 4, 2])
        with mid:
            avg_stats = get_average_stats_by_type(df)
            
            fig4, ax4 = plt.subplots(figsize=(8, 6))
            colors4 = get_pokedex_colors(len(avg_stats))
            bars = ax4.barh(range(len(avg_stats)), avg_stats.values, color=colors4)
            ax4.set_yticks(range(len(avg_stats)))
            ax4.set_yticklabels(avg_stats.index)
            ax4.set_xlabel('Average Total Stats', fontsize=8, fontweight='bold')
            ax4.set_ylabel('Pokemon Type', fontsize=8, fontweight='bold')
            ax4.set_title('Average Power Level by Pokemon Type', fontsize=10, fontweight='bold')
        
            #Adding value labels to the ends of the bars
            for i, v in enumerate(avg_stats.values):
                rounded_val = int(round(v))
                ax4.text(v + 3, i, f'{rounded_val}', va='center', fontweight='bold', fontsize=8)
            ax4.invert_yaxis()
            ax4.grid(axis='x', alpha=0.3, linestyle='--')
            st.pyplot(fig4)
    elif trend_mode == "Stat Distribution by Type":
        #SUB-SECTION: Stat Distribution by Type
        st.header("Stat Distribution by Pokémon Type")
        st.write("Box plots showing how a chosen stat is distributed within each type, sorted by median — reveals both average strength and variance.")
 
        stat_choice_box = st.segmented_control("Select Stat", ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed'], default='Total', key="box_stat")
 
        box_data = get_stat_distribution_by_type(df, stat_choice_box)
        types_ordered = list(box_data.keys())
        values_ordered = [box_data[t] for t in types_ordered]
        n_types = len(types_ordered)
 
        col_box, col_box_insights = st.columns([3, 1])
 
        with col_box:
            fig_box, ax_box = plt.subplots(figsize=(14, 6))
            bp = ax_box.boxplot(
                values_ordered, vert=True, patch_artist=True,
                positions=range(n_types), widths=0.55,
                medianprops=dict(color='white', linewidth=2),
                whiskerprops=dict(color='#555555'),
                capprops=dict(color='#555555'),
                flierprops=dict(marker='o', markersize=3,
                                markerfacecolor='#aaaaaa', alpha=0.5),
            )
            colors_box = get_pokedex_colors(n_types)
            for patch, color in zip(bp['boxes'], colors_box):
                patch.set_facecolor(color)
                patch.set_alpha(0.85)
 
            ax_box.set_xticks(range(n_types))
            ax_box.set_xticklabels(types_ordered, rotation=40, ha='right',
                                   fontsize=9, fontweight='bold')
            ax_box.set_ylabel(stat_choice_box, fontweight='bold', fontsize=11)
            ax_box.set_title(f"{stat_choice_box} Distribution by Primary Type "
                             f"(sorted by median)", fontsize=11, fontweight='bold')
            ax_box.grid(axis='y', alpha=0.3, linestyle='--')
            plt.tight_layout()
            st.pyplot(fig_box)
 
        with col_box_insights:
            st.subheader("How to read this")
            st.write(
                "Each box spans the **middle 50%** of values for that type "
                "(the interquartile range). The **white line** is the median. "
                "Whiskers extend to 1.5× the IQR, and dots beyond that are outliers."
            )
            st.markdown("A **tall box** means high variance — that type contains "
                        "both weak and strong Pokemon. A **short box** means "
                        "most of that type's Pokemon cluster around a similar value.")
            st.subheader("Select Facts")
            st.markdown( f"- The 3 weakest types are Bug, Normal, and Grass.")
            st.markdown( f"- The 3 strongest types are Dragon, Steel, and Psychic (Due to variance).")
            st.markdown( f"- The 3 most varied types are Psychic, Bug, and Fairy.")
            st.markdown( f"- The 3 most consistent are Fighting, Flying, and Electric.")
            st.markdown( f"- Stellar's single white line is Stellar Terapagos.")
 
    elif trend_mode == "Type Composition by Region":
        st.header("Type Composition by Region")
        st.write("What percentage of each generation's Pokémon belong to each type?")
 
        comp_df = get_type_composition_by_region(df)
 
        col_heatmap, col_insights = st.columns([3, 1])
 
        with col_heatmap:
            fig_heat, ax_heat = plt.subplots(figsize=(14, 5))
            sns.heatmap(
                comp_df, ax=ax_heat, cmap='YlOrRd', annot=True, fmt='.1f',
                annot_kws={'size': 7}, linewidths=0.4, linecolor='#cccccc',
                cbar_kws={'label': '% of generation', 'shrink': 0.6},
            )
            ax_heat.set_title("Pokemon Type Composition by Generation (%)", fontsize=12, fontweight='bold', pad=12)
            ax_heat.set_xlabel("Primary Type", fontweight='bold', fontsize=10)
            ax_heat.set_ylabel("Generation", fontweight='bold', fontsize=10)
            ax_heat.set_xticklabels(ax_heat.get_xticklabels(), rotation=40, ha='right', fontsize=8)
            ax_heat.set_yticklabels(ax_heat.get_yticklabels(), rotation=0, fontsize=8)
            plt.tight_layout()
            st.pyplot(fig_heat)
 
        with col_insights:
            st.subheader("How to read this")
            st.write(
                "Each cell is the **% of that generation's Pokemon** with that primary type. Darker = higher share. Rows sum to ~100%."
            )
 
            #Peak generation per type (only types that ever hit 10%+)
            notable_peaks = []
            for t in comp_df.columns:
                peak_val = comp_df[t].max()
                peak_gen = comp_df[t].idxmax()
                if peak_val >= 10.0:
                    parts = peak_gen.split("\n")
                    gen_label = parts[0] + (" (" + parts[1] + ")" if len(parts) > 1 else "")
                    notable_peaks.append((t, gen_label, peak_val))
            notable_peaks.sort(key=lambda x: x[2], reverse=True)
 
            st.markdown("**Type peaks of a Generation (10%+)**")
            for t, gen_label, val in notable_peaks:
                st.markdown(f"- **{t}** peaked in {gen_label} at `{val:.1f}%`")

    else:
        #SUB-SECTION: Base Stat Averages by Generation
        st.header("Base Stat Averages by Generation")
        st.write("Average stats across all Pokémon in each generation, displayed as radar charts.")

        base_only = st.toggle(
            "Base forms only (exclude Mega Evolutions, regional variants, etc.)",
            value=True,
            help="Alternate forms share their original Dex ID, so they skew the "
                 "averages for the generation they were introduced in — not the "
                 "generation the form was added. Enable this for a fairer comparison."
        )
        st.caption(
            "Showing base forms only - Mega Evolutions and regional variants excluded."
            if base_only else
            "Showing all entries including Mega Evolutions and alternate forms."
        )

        STATS      = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
        gen_data   = get_generation_avg_stats(df, base_only=base_only)
        #Shared radial axis max — round up the highest single average to a  ceiling
        all_vals   = [v for _, _, avg in gen_data for v in avg.values()]
        axis_max   = int(round(max(all_vals) / 10 + 0.5) * 10) + 10

        #Pokedex gradient colour per generation
        gen_colors = get_pokedex_colors(9)

        #3 x 3 grid of each visualisation
        for row_idx in range(3):
            cols = st.columns(3)
            for col_idx in range(3):
                gen_idx = row_idx * 3 + col_idx
                gen_num, region, avg = gen_data[gen_idx]

                with cols[col_idx]:
                    st.markdown(
                        f"<h4 style='text-align:center; margin-bottom:0;'>"
                        f"Gen {gen_num} — {region}</h4>",
                        unsafe_allow_html=True
                    )

                    values = [avg[s] for s in STATS]
                    #Close the loop for the filled polygon
                    values_closed = values + [values[0]]
                    stats_closed  = STATS  + [STATS[0]]

                    r, g, b, _ = gen_colors[gen_idx]
                    hex_color   = '#{:02x}{:02x}{:02x}'.format(
                        int(r * 255), int(g * 255), int(b * 255)
                    )

                    fig_gen = go.Figure()
                    rgba_fill = f'rgba({int(r*255)},{int(g*255)},{int(b*255)},0.3)'
                    fig_gen.add_trace(go.Scatterpolar(
                        r=values_closed,
                        theta=stats_closed,
                        fill='toself',
                        fillcolor=rgba_fill,
                        line=dict(color=hex_color, width=2),
                        name=region,
                        hovertemplate='%{theta}: %{r:.1f}<extra></extra>',
                    ))
                    fig_gen.update_layout(
                        polar=dict(
                            bgcolor='lightblue',
                            radialaxis=dict(
                                visible=True,
                                range=[0, axis_max],
                                tickfont=dict(size=7),
                                tickangle=45,
                            ),
                            angularaxis=dict(tickfont=dict(size=8)),
                        ),
                        showlegend=False,
                        margin=dict(l=30, r=30, t=10, b=10),
                        height=280,
                    )
                    st.plotly_chart(fig_gen, use_container_width=True)

                    #Stat value summary beneath each chart
                    stat_lines = "  |  ".join(
                        f"{s}: {avg[s]}" for s in STATS
                    )
                    st.markdown(
                        f"<p style='text-align:center; font-size:11px; "
                        f"line-height:1.8;'>{stat_lines}</p>",
                        unsafe_allow_html=True
                    )

#MAIN TAB 3: RELATIONSHIPS
with main_tabs[2]:
    relationship_mode = st.segmented_control("View Relationships By:", ["Stat Correlation", "Pokemon Comparison Chart"], default="Stat Correlation", key="relationship_mode")

    if relationship_mode == "Stat Correlation":
        st.header("Stat Correlations")

        col_chart, col_explain = st.columns([3, 2])

        with col_chart:
            corr = get_stat_correlation(df)
            fig_corr, ax_corr = plt.subplots()
            sns.heatmap(corr, annot=True, cmap='RdYlGn', ax=ax_corr,
                       linewidths=0.5, linecolor='lightgrey')
            st.pyplot(fig_corr)

        with col_explain:
            st.subheader("How to read this chart")
            st.write(
                "Each cell shows how strongly two stats move together across all Pokémon. "
                "Values run from **-1** (opposite extremes) through **0** (no relationship) "
                "to **+1** (perfectly in step). The colour reinforces this — "
                "green means a positive correlation, red means a negative one."
            )
            st.markdown("---")
            st.subheader("Notable relationships")

            # Derive the strongest off-diagonal pairs directly from the data
            # so the explanations always reflect the actual numbers.
            corr_pairs = (
                corr.where(~pd.DataFrame(
                    [[i == j for j in range(len(corr.columns))]
                     for i in range(len(corr.columns))],
                    index=corr.index, columns=corr.columns
                ))
                .stack()
                .drop_duplicates()
            )
            top_pos = corr_pairs.nlargest(3)
            top_neg = corr_pairs.nsmallest(3)

            st.markdown("**Strongest positive correlations**")
            for (a, b), val in top_pos.items():
                st.markdown(f"- **{a} ↔ {b}**: `{val:.2f}` — Pokémon with high {a} tend to also have high {b}.")

            st.markdown("**Strongest negative correlations**")
            for (a, b), val in top_neg.items():
                st.markdown(f"- **{a} ↔ {b}**: `{val:.2f}` — Pokémon that excel in {a} often sacrifice {b}.")
    else:
        st.header("Stat Radar Comparison")
        st.write("Compare the stat 'shape' of two Pokémon.")
        
        # Pokemon selection in two columns
        selection_col1, selection_col2 = st.columns(2)

        _names = list(df['Name'].unique())

        with selection_col1:
            _search1 = st.text_input("Search Pokémon 1", placeholder="Type to filter…", key="search_p1")
            _filtered1 = [n for n in _names if _search1.lower() in n.lower()] if _search1 else _names
            _p1_current = st.session_state.get("radar_p1")
            _p1_default = _filtered1.index(_p1_current) if _p1_current in _filtered1 else 0
            p1 = st.selectbox("Select Pokémon 1", _filtered1, index=_p1_default, key="radar_p1", label_visibility="collapsed")

        with selection_col2:
            _search2 = st.text_input("Search Pokémon 2", placeholder="Type to filter…", key="search_p2")
            _filtered2 = [n for n in _names if _search2.lower() in n.lower()] if _search2 else _names
            _p2_current = st.session_state.get("radar_p2")
            _p2_default = _filtered2.index(_p2_current) if _p2_current in _filtered2 else 0
            p2 = st.selectbox("Select Pokémon 2", _filtered2, index=_p2_default, key="radar_p2", label_visibility="collapsed")

        #Create the radar chart function
        def create_radar(name1, name2):
            categories = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
            fig = go.Figure()

            for name, color in [(name1, '#EF5350'), (name2, '#42A5F5')]:
                stats = get_pokemon_stats(df, name, categories)
                stats.append(stats[0])  #Close the loop
                fig.add_trace(go.Scatterpolar(
                    r=stats, 
                    theta=categories + [categories[0]],
                    fill='toself', 
                    name=name, 
                    line_color=color
                ))

            fig.update_layout(
                polar=dict(
                    bgcolor="lightblue",
                    radialaxis=dict(visible=True, range=[0, 255]),
                ),
                showlegend=True,
                height=500,
                margin=dict(l=80, r=80, t=20, b=20),
                font=dict(color='black'),
            )

            return fig

        #Left Sprite/Radar Chart/Right Sprite
        pad_left, chart_middle, pad_right = st.columns([1, 3, 1])

        with chart_middle:
            st.plotly_chart(create_radar(p1, p2), use_container_width=True)

        #Sprites in a balanced two-column strip below the chart.
        #Each column gets exactly half the page, so both sprites scale identically
        sprite_col1, sprite_col2 = st.columns(2)

        sprite1_path = get_sprite_path(p1, df)
        sprite2_path = get_sprite_path(p2, df)
        type1_sprites = get_type_sprite_paths(p1, df)
        type2_sprites = get_type_sprite_paths(p2, df)


        def render_pokemon_panel(name, sprite_path, type_sprites, shared_abilities, shared_moves):
            st.markdown(f"<h3 style='text-align: center;'>{name}</h3>", unsafe_allow_html=True)

            #Pokémon sprite so it doesn't stretch on wide screens
            _, img_area, _ = st.columns([1, 1, 1])
            with img_area:
                if sprite_path:
                    st.image(sprite_path, use_container_width=True)
                else:
                    st.caption("Sprite not available")

            #Type badge(s) centred beneath the sprite.
            #For a single type: one badge centred. For dual types: two badges side by side.
            if type_sprites:
                num_types = len(type_sprites)
                #Narrow padding columns either side of the badges
                if num_types == 1:
                    _, badge_col, _ = st.columns([1.5, 1, 1.5])
                    badge_cols = [badge_col]
                else:
                    _, b1, b2, _ = st.columns([0.75, 0.75, 0.75, 0.75])
                    badge_cols = [b1, b2]

                for (type_name, type_path), col in zip(type_sprites, badge_cols):
                    with col:
                        st.image(type_path)
            #Abilities
            normal_abs, hidden_ab = get_pokemon_abilities(df, name)
            all_abs = normal_abs + ([f"{hidden_ab} (Hidden)"] if hidden_ab else [])
 
            st.markdown("<p style='text-align:center; font-size:0.75rem; "
                        "color:#aaaaaa; margin:8px 0 2px 0; letter-spacing:1px;'>"
                        "ABILITIES</p>", unsafe_allow_html=True)
 
            ab_items_html = ''.join(
                "<div style='color:{c};padding:2px 0;font-size:0.82rem;"
                "font-weight:600;text-align:center;'>{ab}</div>".format(
                    c="#4caf50" if ab.replace(" (Hidden)","").strip() in shared_abilities else "#e0e0e0",
                    ab=ab
                )
                for ab in all_abs
            ) if all_abs else "<div style='color:#666;font-size:0.8rem;text-align:center;'>No ability data</div>"
 
            st.markdown(
                "<div style='"
                "background-color:#242424;"
                "border:8px solid #d0d0d0;"
                "border-radius:6px;"
                "box-shadow:0 0 0 2px #c1c1c1,inset 0 2px 6px rgba(0,0,0,0.35);"
                "padding:12px 12px 14px 12px;"
                "font-family:monospace;"
                "'>"
                + ab_items_html +
                "</div>",
                unsafe_allow_html=True
            )
            #Moves 
            move_data = get_pokemon_moves(learnsets_df, name)
            METHOD_LABELS = {
                'level-up':    'LEVEL UP',
                'machine':     'TM',
                'egg':         'EGG',
                'tutor':       'TUTOR',
                'form-change': 'FORM CHANGE',
            }
            METHOD_ORDER = ['level-up', 'machine', 'egg', 'tutor', 'form-change']
            active_methods = [m for m in METHOD_ORDER if move_data.get(m)]
            if active_methods:
                st.markdown(
                    "<p style='text-align:center; font-size:1rem; "
                    "margin:12px 0 6px 0; letter-spacing:1px;'>"
                    "MOVES</p>",
                    unsafe_allow_html=True
                )
                method_cols = st.columns(len(active_methods))
                for col, method in zip(method_cols, active_methods):
                    with col:
                        label  = METHOD_LABELS.get(method, method.upper())
                        moves_list = move_data[method]
                        chunk_size = 15
                        chunks = [moves_list[i:i+chunk_size] for i in range(0, len(moves_list), chunk_size)]
                        chunks_html = ""
                        _comp_mv_lookup = build_moves_lookup(moves_meta_df)
                        # Build a level lookup for this Pokemon's level-up moves
                        _comp_ls = learnsets_df[
                            (learnsets_df['pokemon_name'] == name) &
                            (learnsets_df['learn_method'] == 'level-up')
                        ].set_index('move_name')['level']
                        for chunk in chunks:
                            def _comp_tip(mv):
                                colour = '#4caf50' if mv in shared_moves else '#e0e0e0'
                                info   = _comp_mv_lookup.get(mv, {})
                                mtype  = info.get('type', '--')
                                dclass = info.get('damage_class', '--')
                                pp     = info.get('pp', '--')
                                pwr    = info.get('power', '--')
                                acc    = info.get('accuracy', '--')
                                effect = info.get('effect', '')
                                lvl    = _comp_ls.get(mv)
                                level_str = f'<br>Learned at: Lv. {int(lvl)}' if lvl is not None and pd.notna(lvl) and int(lvl) > 0 else ''
                                tip = (
                                    f'<b>{mv}</b><br>'
                                    f'{mtype} &middot; {dclass}<br>'
                                    f'PP: {pp} &nbsp; Pwr: {pwr} &nbsp; Acc: {acc}'
                                    + level_str
                                    + (f'<br><i>{effect}</i>' if effect else '')
                                )
                                return (
                                    f"<span class='move-tip' style='display:block;'>"
                                    f"<div style='color:{colour};padding:1px 0;"
                                    f"font-size:0.6rem;word-break:break-word;'>{mv}</div>"
                                    f"<span class='tip-box'>{tip}</span>"
                                    f"</span>"
                                )
                            chunk_items = ''.join(_comp_tip(mv) for mv in chunk)
                            is_last = (chunk == chunks[-1])
                            border = "" if is_last else "border-right:1px solid #444;padding-right:1px;"
                            chunks_html += (
                                f"<div style='flex:1;min-width:0;{border}'>"
                                + chunk_items +
                                "</div>"
                            )
                        card = (
                            "<div style='"
                            "background-color:#242424;"
                            "border:8px solid #d0d0d0;"
                            "border-radius:6px;"
                            "box-shadow:0 0 0 2px #c1c1c1,inset 0 2px 6px rgba(0,0,0,0.35);"
                            "padding:12px 12px 14px 12px;"
                            "font-family:monospace;"
                            "'>"
                            f"<p style='font-size:0.7rem;color:#aaaaaa;margin:0 0 8px 0;"
                            f"letter-spacing:2px;text-transform:uppercase;'> ▶ {label}</p>"
                            "<div style='display:flex;flex-direction:row;gap:10px;'>"
                            + chunks_html +
                            "</div>"
                            "</div>"
                        )
                        st.markdown(card, unsafe_allow_html=True)
        #Compute shared abilities (normal + hidden combined) for colour-coding
        def _ability_set(name):
            normal, hidden = get_pokemon_abilities(df, name)
            return set(normal) | ({hidden} if hidden else set())
 
        shared_abilities = _ability_set(p1) & _ability_set(p2)
 
        def _move_set(name):
            data = get_pokemon_moves(learnsets_df, name)
            return {m for moves in data.values() for m in moves}
 
        shared_moves = _move_set(p1) & _move_set(p2)
 
        with sprite_col1:
            render_pokemon_panel(p1, sprite1_path, type1_sprites, shared_abilities, shared_moves)
 
        with sprite_col2:
            render_pokemon_panel(p2, sprite2_path, type2_sprites, shared_abilities, shared_moves)

#MAIN TAB 4: MACHINE LEARNING
with main_tabs[3]:
    ml_mode = st.segmented_control("View Machine Learning Method By:", ["K-Means", "DBSCAN"], default="K-Means", key="ml_mode")
    if ml_mode == "K-Means":
        st.header("Pokémon Archetype Clustering - K-Means")
        st.write("Use Machine Learning to group Pokémon based on Competitive Archetypes with K-Means = 7.")
        @st.fragment
        def clustering_analysis():

            selected_features = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
            k_val = 8

            #Clustering
            df_clustered = perform_clustering(df, selected_features, k_val)

            cluster_summary = get_cluster_summary(df_clustered, selected_features)
            archetype_labels = label_archetypes(cluster_summary, selected_features)
            df_clustered['Archetype'] = df_clustered['Cluster'].map(archetype_labels)

            #Compute the archetype axes
            df_plot = add_archetype_axes(df_clustered, selected_features).copy()
            rng = np.random.default_rng(seed=42)
            jitter_scale = 4
            df_plot['_axis_x'] = df_plot['_axis_x'] + rng.uniform(-jitter_scale, jitter_scale, size=len(df_plot))
            df_plot['_axis_y'] = df_plot['_axis_y'] + rng.uniform(-jitter_scale, jitter_scale, size=len(df_plot))

            st.markdown("---")
            st.subheader("Archetype Map")
            st.caption(
                "**X-axis**: Attack - Sp. Atk — left = Special Attacker, right = Physical Attacker  |  "
                "**Y-axis**: Speed - avg(Def, Sp. Def, HP) — bottom = Bulky Wall, top = Fast Sweeper"
            )

            col_left, col_right = st.columns([3, 2])
            with col_left:
                fig_km = px.scatter(
                    df_plot,
                    x='_axis_x',
                    y='_axis_y',
                    color='Archetype',
                    hover_name='Name',
                    hover_data={s: True for s in selected_features} | {
                        '_axis_x': False, '_axis_y': False
                    },
                    labels={
                        '_axis_x': '← Special Attacker  |  Physical Attacker →',
                        '_axis_y': '← Bulky Wall  |  Fast Sweeper →',
                        'color':   'Archetype',
                    },
                    template="plotly_white",
                    color_discrete_sequence=px.colors.qualitative.Safe,
                )
                fig_km.update_traces(marker=dict(size=8, opacity=0.7,
                                                 line=dict(width=1, color='DarkSlateGrey')))

                #Quadrant reference lines at zero
                fig_km.add_hline(y=0, line_dash='dot', line_color='grey', opacity=0.4)
                fig_km.add_vline(x=0, line_dash='dot', line_color='grey', opacity=0.4)

                #Quadrant corner annotations
                quadrant_notes = [
                    (0.97, 0.97, 'right', 'top',    'Phys. Sweeper'),
                    (0.03, 0.97, 'left',  'top',    'Spec. Sweeper'),
                    (0.97, 0.03, 'right', 'bottom', 'Phys. Wall'),
                    (0.03, 0.03, 'left',  'bottom', 'Spec. Wall'),
                ]
                for xr, yr, xanchor, yanchor, txt in quadrant_notes:
                    fig_km.add_annotation(
                        xref='paper', yref='paper',
                        x=xr, y=yr,
                        text=f"<i>{txt}</i>",
                        showarrow=False,
                        font=dict(size=10, color='grey'),
                        xanchor=xanchor, yanchor=yanchor,
                    )

                st.plotly_chart(fig_km, use_container_width=True)
                st.markdown("---")
                st.write("This is largely left as a first example foray into K-Means clustering, whilst the other tabs in the Machine Learning section focus on alternative solutions.")

            with col_right:
                st.write("**Archetype Average Stats**")
                display_summary = cluster_summary.copy()
                display_summary.index = [archetype_labels[i] for i in display_summary.index]
                display_summary = display_summary.groupby(display_summary.index).mean().round(0).astype(int)
                st.dataframe(display_summary)
                st.markdown("---")
                st.subheader("Notable Outliers")
                st.write(
                    "K-means has limitations when it comes to clustering similar Pokémon and outliers. "
                    "Pokémon such as Blissey, Eternatus Eternamax, and Shuckle, are emblematic of this. "
                    "Having very high stats in multiple fields, they tend to distort the averages "
                    "or otherwise are grouped incorrectly. "
                    "To solve this would require some significant clustering i.e. 12, "
                    "an alternative algorithm to group the Pokémon, or new archetypes that are of a more "
                    "generalised form than splitting on Special/Physical axis."

                )

        clustering_analysis()
    elif ml_mode == "DBSCAN":
        st.header("Pokémon Archetype Clustering - DBSCAN")
        st.write(
            "This takes a hybrid approach: **DBSCAN** first identifies statistical outliers in raw stat space (Pokémon whose stat profiles are too extreme or unusual to belong to any mainstream archetype), then **K-Means** clusters the remaining Pokémon into 6 competitive archetypes on a clean dataset. "
            "This gives more accurate archetype assignments than pure K-Means because extreme outliers like Blissey, Shuckle, and Eternatus no longer distort the cluster centroids."
        )

        @st.fragment
        def dbscan_analysis():
            selected_features = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']

            # Run the two-phase pipeline
            df_db, core_summary = perform_dbscan(df, selected_features, eps=2, min_samples=8)
            db_labels = label_archetypes_dbscan(core_summary, selected_features)

            #"Rescue Step"
            #Basically, after the two-phases are run, there are still some significant outliers but they could still be assigned to specific roles
            #Thus, rule-based absolute value determination allows the handling of specific edge cases
            RESCUE_THRESHOLD = 0.38 #Max single stat share calculated by comparing pokemon stat totals like Blissey with Mega Aggron
            #Above a ratio of 0.38, you get the defining stat of the pokemon (Usually HP)

            _ICONS = {
                'Physical Sweeper': '⚔️', 'Special Sweeper': '✨',
                'Physical Wall':    '🛡️', 'Special Wall':    '🔮',
                'Bulky Attacker':   '💪', 'Bulky Special Attacker': '🔯',
            }

            def _rescue_archetype(row):
                """Return 'icon archetype' if the Pokemon fits a clear role, else None."""
                atk=row['Attack']; spatk=row['Sp. Atk']; spd=row['Speed']
                dfn=row['Defense']; spdef=row['Sp. Def']
                if spd >= 100:
                    arch = 'Physical Sweeper' if atk >= spatk else 'Special Sweeper'
                elif atk >= 100 and dfn >= 120:
                    arch = 'Bulky Attacker'
                elif spatk >= 110 and spdef >= 120:
                    arch = 'Bulky Special Attacker'
                elif dfn > spdef and dfn >= 120:
                    arch = 'Physical Wall'
                elif spdef >= dfn and spdef >= 120:
                    arch = 'Special Wall'
                else:
                    return None
                return f"{_ICONS[arch]} {arch}"

            _totals    = df_db[selected_features].sum(axis=1).replace(0, 1)
            _max_ratio = df_db[selected_features].div(_totals, axis=0).max(axis=1)

            def _assign_archetype(row):
                if not row['Is_Outlier']:
                    return db_labels.get(int(row['Cluster']), 'Unknown')
                if _max_ratio[row.name] < RESCUE_THRESHOLD:
                    rescued = _rescue_archetype(row)
                    if rescued:
                        return rescued
                return '⚠️ Outlier'
            #Map labels; outliers get their own display label
            df_db['Archetype'] = df_db.apply(_assign_archetype, axis=1)

            #Compute scatter axes (same formula as K-Means tab)
            df_plot = add_archetype_axes(df_db, selected_features).copy()
            rng = np.random.default_rng(seed=42)
            jitter_scale = 4
            df_plot['_axis_x'] = df_plot['_axis_x'] + rng.uniform(-jitter_scale, jitter_scale, size=len(df_plot))
            df_plot['_axis_y'] = df_plot['_axis_y'] + rng.uniform(-jitter_scale, jitter_scale, size=len(df_plot))

            #Separate outliers for distinct marker styling
            df_core    = df_plot[df_plot['Archetype'] != '⚠️ Outlier']
            df_outlier = df_plot[df_plot['Archetype'] == '⚠️ Outlier']

            n_outliers = df_outlier.shape[0]
            n_core     = df_core.shape[0]

            n_outliers = df_outlier.shape[0]
            n_core     = df_core.shape[0]

            st.markdown("---")
            st.subheader("Archetype Map")
            st.caption(
                f"**{n_core}** Pokémon assigned to archetypes &nbsp;|&nbsp; "
                f"**{n_outliers}** flagged as outliers &nbsp;|&nbsp; "
                "**X-axis**: Attack - Sp. Atk — left = Special Attacker, right = Physical Attacker  &nbsp;|&nbsp;  "
                "**Y-axis**: Speed - avg(Def, Sp. Def, HP) — bottom = Bulky, top = Fast Sweeper"
            )

            col_left, col_right = st.columns([3, 2])

            with col_left:
                # Colour palette matching K-Means tab (Safe qualitative) for archetypes
                archetype_order = sorted(df_core['Archetype'].unique())
                palette = px.colors.qualitative.Safe[:len(archetype_order)]
                color_map = {arch: col for arch, col in zip(archetype_order, palette)}
                color_map['⚠️ Outlier'] = '#888888'

                fig_db = px.scatter(
                    df_core,
                    x='_axis_x',
                    y='_axis_y',
                    color='Archetype',
                    hover_name='Name',
                    hover_data={s: True for s in selected_features} | {'_axis_x': False, '_axis_y': False},
                    labels={
                        '_axis_x': '← Special Attacker  |  Physical Attacker →',
                        '_axis_y': '← Bulky Wall  |  Fast Sweeper →',
                        'color': 'Archetype',
                    },
                    template="plotly_white",
                    color_discrete_map=color_map,
                    category_orders={'Archetype': archetype_order},
                )
                fig_db.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))

                # Outliers as grey X markers layered on top
                fig_db.add_trace(go.Scatter(
                    x=df_outlier['_axis_x'],
                    y=df_outlier['_axis_y'],
                    mode='markers',
                    name='⚠️ Outlier',
                    marker=dict(symbol='x', size=9, color='#888888', opacity=0.6,
                                line=dict(width=1.5, color='#555555')),
                    hovertemplate=(
                        '<b>%{customdata[0]}</b><br>'
                        'HP=%{customdata[1]}  Atk=%{customdata[2]}  Def=%{customdata[3]}<br>'
                        'SpA=%{customdata[4]}  SpD=%{customdata[5]}  Spe=%{customdata[6]}'
                        '<extra>⚠️ Outlier</extra>'
                    ),
                    customdata=df_outlier[['Name'] + selected_features].values,
                ))

                # Quadrant reference lines
                fig_db.add_hline(y=0, line_dash='dot', line_color='grey', opacity=0.4)
                fig_db.add_vline(x=0, line_dash='dot', line_color='grey', opacity=0.4)

                for xr, yr, xa, ya, txt in [
                    (0.97, 0.97, 'right', 'top',    'Phys. Sweeper'),
                    (0.03, 0.97, 'left',  'top',    'Spec. Sweeper'),
                    (0.97, 0.03, 'right', 'bottom', 'Phys. Wall'),
                    (0.03, 0.03, 'left',  'bottom', 'Spec. Wall'),
                ]:
                    fig_db.add_annotation(
                        xref='paper', yref='paper', x=xr, y=yr,
                        text=f"<i>{txt}</i>", showarrow=False,
                        font=dict(size=10, color='grey'), xanchor=xa, yanchor=ya,
                    )

                st.plotly_chart(fig_db, use_container_width=True)

            with col_right:
                #Archetype average stats
                st.write("**Archetype Average Stats**")
                display_summary = core_summary.copy()
                display_summary.index = [db_labels[i] for i in display_summary.index]
                display_summary = display_summary.groupby(display_summary.index).mean().round(0).astype(int)
                st.dataframe(display_summary, use_container_width=True)

                st.markdown("---")

                #Outlier table
                st.write(f"**Outliers** ({n_outliers} Pokémon)")
                st.caption(
                    "These Pokémon have stat profiles too extreme or unusual to fit the archetypes. "
                    "Removing them before clustering means they no longer skew the archetype centroids. "
                    "They could basically be classified as their own class of Pokémon - General Tanks. "
                    "Depending on other ways to query/sort the Pokémon data, you could assign more Pokémon to this set too."
                )
                outlier_rows = df_db[df_db['Archetype'] == '⚠️ Outlier'].copy()
                outlier_rows['Name'] = outlier_rows['Name'].str.replace('\n', ' ', regex=False)
                # Sort by Total descending so legendaries appear first
                outlier_display = (
                    outlier_rows[['Name'] + selected_features + ['Total']]
                    .sort_values('Total', ascending=False)
                    .reset_index(drop=True)
                )
                st.dataframe(outlier_display, use_container_width=True)

        dbscan_analysis()
#MAIN TAB 5: POKEDEX LOOKUP
with main_tabs[4]:
    st.header("Pokédex Lookup")
    st.write("Select a Pokemon to view its stats, types, abilities, design origin, and FAQ.")
    st.write("Hover over moves to see its information.")
    try:
        _api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        _api_key = ""
 
    #Pokemon selector with live search filter
    _lookup_names = list(df['Name'].unique())
    lk_col1, _ = st.columns([1, 3])
    with lk_col1:
        _lk_search   = st.text_input("Search Pokemon", placeholder="Type to filter...", key="lk_search")
        _lk_filtered = [n for n in _lookup_names if _lk_search.lower() in n.lower()] if _lk_search else _lookup_names
        _lk_current  = st.session_state.get("lk_pokemon")
        _lk_default  = _lk_filtered.index(_lk_current) if _lk_current in _lk_filtered else 0
        lk_pokemon   = st.selectbox("Select Pokemon", _lk_filtered, index=_lk_default, key="lk_pokemon", label_visibility="collapsed")
 
    st.divider()
 
    #Main layout: left = sprite/types/abilities/stats, right = radar chart
    lk_left, lk_right = st.columns([1, 2])
 
    with lk_left:
        _name_display = lk_pokemon.replace(chr(10), ' ')
        st.markdown(
            f"<h2 style='text-align:center;'>{_name_display}</h2>",
            unsafe_allow_html=True
        )
 
        lk_sprite = get_sprite_path(lk_pokemon, df)
        _, lk_img, _ = st.columns([1, 2, 1])
        with lk_img:
            if lk_sprite:
                st.image(lk_sprite, use_container_width=True)
            else:
                st.caption("Sprite not available")
 
        lk_types = get_type_sprite_paths(lk_pokemon, df)
        if lk_types:
            if len(lk_types) == 1:
                _, tb, _ = st.columns([1.5, 1, 1.5])
                with tb: st.image(lk_types[0][1])
            else:
                _, tb1, tb2, _ = st.columns([0.75, 0.75, 0.75, 0.75])
                with tb1: st.image(lk_types[0][1])
                with tb2: st.image(lk_types[1][1])
 
        lk_normal_abs, lk_hidden_ab = get_pokemon_abilities(df, lk_pokemon)
        lk_all_abs = lk_normal_abs + ([f"{lk_hidden_ab} (Hidden)"] if lk_hidden_ab else [])
        if lk_all_abs:
            st.markdown(
                "<p style='text-align:center; font-size:0.75rem; color:#aaaaaa;"
                " margin:10px 0 4px 0; letter-spacing:1px;'>ABILITIES</p>",
                unsafe_allow_html=True
            )
            for ab in lk_all_abs:
                st.markdown(
                    f"<p style='text-align:center; margin:2px 0;"
                    f" font-size:0.95rem; font-weight:600;'>{ab}</p>",
                    unsafe_allow_html=True
                )
 
        _stat_cols = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Total']
        lk_row = df[df['Name'] == lk_pokemon]
        if not lk_row.empty:
            st.markdown(
                "<p style='text-align:center; font-size:0.75rem; color:#aaaaaa;"
                " margin:14px 0 4px 0; letter-spacing:1px;'>BASE STATS</p>",
                unsafe_allow_html=True
            )
            _sr = lk_row.iloc[0]
            for sc in _stat_cols:
                val = _sr.get(sc, 'N/A')
                st.markdown(
                    f"<p style='text-align:center; margin:1px 0;"
                    f" font-size:0.88rem;'><b>{sc}</b>: {val}</p>",
                    unsafe_allow_html=True
                )
 
    with lk_right:
        lk_categories = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
        lk_stats = get_pokemon_stats(df, lk_pokemon, lk_categories)
        if lk_stats:
            lk_stats_closed = lk_stats + [lk_stats[0]]
            fig_lk = go.Figure()
            fig_lk.add_trace(go.Scatterpolar(
                r=lk_stats_closed,
                theta=lk_categories + [lk_categories[0]],
                fill='toself',
                fillcolor='rgba(239,83,80,0.3)',
                line=dict(color='#EF5350', width=2),
                hovertemplate='%{theta}: %{r}<extra></extra>',
            ))
            fig_lk.update_layout(
                polar=dict(
                    bgcolor='lightblue',
                    radialaxis=dict(visible=True, range=[0, 255]),
                ),
                showlegend=False,
                height=420,
                margin=dict(l=60, r=60, t=30, b=30),
                font=dict(color='black'),
            )
            st.plotly_chart(fig_lk, use_container_width=True)
 
    st.divider()
 
    #Moves section
    def _make_move_tip(mv, pokemon_name, mv_lookup, ls_rows):
        info  = mv_lookup.get(mv, {})
        mtype = info.get('type', '--')
        dclass= info.get('damage_class', '--')
        pp    = info.get('pp', '--')
        pwr   = info.get('power', '--')
        acc   = info.get('accuracy', '--')
        effect= info.get('effect', '')
        # Level learned (only meaningful for level-up moves)
        level_str = ''
        if mv in ls_rows.index:
            row = ls_rows.loc[mv]
            # loc returns a Series for a single match, DataFrame for multiple
            if isinstance(row, pd.Series):
                lvl = row['level']
            else:
                # Multiple rows — pick the level-up entry if present, else first
                lu = row[row['learn_method'] == 'level-up']
                lvl = lu.iloc[0]['level'] if not lu.empty else row.iloc[0]['level']
            if pd.notna(lvl) and int(lvl) > 0:
                level_str = f'<br>Learned at: Lv. {int(lvl)}'
        tip_content = (
            f'<b>{mv}</b><br>'
            f'{mtype} &middot; {dclass}<br>'
            f'PP: {pp} &nbsp; PWR: {pwr} &nbsp; ACC: {acc}'
            + level_str +
            (f'<br><i>{effect}</i>' if effect else '')
        )
        return (
            f"<span class='move-tip' style='display:block;'>"
            f"<div style='padding:1px 0;font-size:0.75rem;color:#e0e0e0;word-break:break-word;'>{mv}</div>"
            f"<span class='tip-box'>{tip_content}</span>"
            f"</span>"
        )

    lk_move_data = get_pokemon_moves(learnsets_df, lk_pokemon)
    LK_METHOD_LABELS = {
        'level-up':    'LEVEL UP',
        'machine':     'TM',
        'egg':         'EGG',
        'tutor':       'TUTOR',
        'form-change': 'FORM CHANGE',
    }
    LK_METHOD_ORDER = ['level-up', 'machine', 'egg', 'tutor', 'form-change']
    lk_active_methods = [m for m in LK_METHOD_ORDER if lk_move_data.get(m)]
 
    if lk_active_methods:
        st.markdown(
            f"<h3 style='text-align:left;'>MOVES</h2>",
            unsafe_allow_html=True
        )
        chunk_size = 15
        col_widths = [
            max(1, len(lk_move_data[m]) // chunk_size + (1 if len(lk_move_data[m]) % chunk_size else 0))
            for m in lk_active_methods
        ]
        lk_method_cols = st.columns(col_widths)
        for col, method in zip(lk_method_cols, lk_active_methods):
            with col:
                label = LK_METHOD_LABELS.get(method, method.upper())
                st.markdown(
                    f"<p style='font-size:0.7rem; color:#aaaaaa; margin:0 0 6px 0; "
                    f"letter-spacing:2px; text-transform:uppercase;'>▶ {label}</p>",
                    unsafe_allow_html=True
                )
                moves_list = lk_move_data[method]
                chunks = [moves_list[i:i+chunk_size] for i in range(0, len(moves_list), chunk_size)]
                chunks_html = ""
                for chunk in chunks:
                    mv_lookup = build_moves_lookup(moves_meta_df)
                    ls_rows   = learnsets_df[
                        (learnsets_df['pokemon_name'] == lk_pokemon) &
                        (learnsets_df['move_name'].isin(chunk))
                    ].set_index('move_name')
                    chunk_items = ''.join(
                        _make_move_tip(mv, lk_pokemon, mv_lookup, ls_rows)
                        for mv in chunk
                    )
                    border = "border-right:1px solid #444; padding-right:16px;"
                    chunks_html += (
                        f"<div style='flex:1; min-width:0;"
                        f"overflow-wrap:break-word; {border}'>"
                        + chunk_items
                        + "</div>"
                    )
                st.markdown(
                    "<div style='display:flex; flex-direction:row; gap:16px; "
                    "font-family:monospace;'>"
                    + chunks_html +
                    "</div>",
                    unsafe_allow_html=True
                )

    smogon_key   = _smogon_name(lk_pokemon)
    pokemon_sets = smogon_sets.get(smogon_key, {}) if smogon_key else {}
 
    if pokemon_sets:
        st.divider()
        tier = pokemon_sets.get('tier', '')
        sets = pokemon_sets.get('sets', {})
        st.markdown(
            f"<h3 style='margin-bottom:4px;'>Competitive Sets</h3>"
            f"<p style='font-size:0.8rem;color:#aaaaaa;margin-top:0;'>"
            f"Source: Smogon &nbsp;&middot;&nbsp; Tier: <b>{tier}</b></p>",
            unsafe_allow_html=True
        )
        EV_MAP = {'hp':'HP','atk':'Atk','def':'Def','spa':'SpA','spd':'SpD','spe':'Spe'}
 
        def _flat(val):
            if val is None: return '--'
            if isinstance(val, list): return ' / '.join(str(v) for v in val)
            return str(val)
 
        for set_name, s in sets.items():
            moves_flat = [
                ' / '.join(m) if isinstance(m, list) else m
                for m in s.get('moves', [])
            ]
            item    = _flat(s.get('item'))
            ability = _flat(s.get('ability'))
            nature  = _flat(s.get('nature'))
            tera    = _flat(s.get('teratypes'))
            evs_raw = s.get('evs', {})
            if isinstance(evs_raw, list):
                evs_raw = evs_raw[0]
            ev_str = ' / '.join(
                f"{v} {EV_MAP.get(k,k)}" for k,v in evs_raw.items() if v
            ) if evs_raw else '--'
 
            mv_lookup_comp = build_moves_lookup(moves_meta_df)
            def _tip_or_plain(move_entry):
                #move_entry may be 'Move A / Move B' (slash options)
                #wrap each option individually then rejoin
                parts = [p.strip() for p in move_entry.split(' / ')]
                tipped = ' / '.join(
                    _move_tip_html(p, mv_lookup_comp) if p in mv_lookup_comp else p
                    for p in parts
                )
                return f"<div style='padding:1px 0;'>&#9658; {tipped}</div>"
            moves_html = ''.join(_tip_or_plain(m) for m in moves_flat)
            card = (
                "<div style='background-color:#242424;border:8px solid #d0d0d0;"
                "border-radius:6px;box-shadow:0 0 0 2px #c1c1c1,"
                "inset 0 2px 6px rgba(0,0,0,0.35);padding:14px 18px 16px 18px;"
                "font-family:monospace;margin-bottom:12px;'>"
                f"<p style='font-size:0.8rem;color:#aaaaaa;margin:0 0 10px 0;"
                f"letter-spacing:2px;text-transform:uppercase;'>&#9658; {set_name}</p>"
                "<div style='display:flex;gap:32px;'>"
                "<div style='flex:1;'>"
                "<p style='font-size:0.65rem;color:#777;margin:0 0 4px 0;"
                "letter-spacing:1px;'>MOVES</p>"
                f"<div style='font-size:0.82rem;color:#e0e0e0;'>{moves_html}</div>"
                "</div>"
                "<div style='flex:1;font-size:0.82rem;color:#e0e0e0;'>"
                "<p style='font-size:0.65rem;color:#777;margin:0 0 4px 0;"
                "letter-spacing:1px;'>DETAILS</p>"
                f"<div><span style='color:#aaa;'>Item: </span>{item}</div>"
                f"<div><span style='color:#aaa;'>Ability: </span>{ability}</div>"
                f"<div><span style='color:#aaa;'>Nature: </span>{nature}</div>"
                f"<div><span style='color:#aaa;'>Tera: </span>{tera}</div>"
                f"<div><span style='color:#aaa;'>EVs: </span>{ev_str}</div>"
                "</div></div></div>"
            )
            st.markdown(card, unsafe_allow_html=True)
    elif smogon_key:
        st.divider()
        st.markdown(
            f"<p style='color:#888;font-size:0.9rem;'>No competitive sets found for "
            f"{lk_pokemon.replace(chr(10),' ')} — this Pokémon may not have a dedicated "
            f"Smogon tier set.</p>",
            unsafe_allow_html=True
        )
    st.divider()
    #AI-generated Design Origin and FAQ AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    if not _api_key:
        st.warning( "No Anthropic API key found.")
    else:
        _origin_key = f"_lk_origin_{lk_pokemon}"
        _faq_key    = f"_lk_faq_{lk_pokemon}"
 
        if _origin_key not in st.session_state or _faq_key not in st.session_state:
            _name_clean = lk_pokemon.replace(chr(10), ' ')
            with st.spinner(f"Looking up {_name_clean}..."):
 
                def _call_claude(prompt):
                    client = _anthropic.Anthropic(api_key=_api_key)
                    message = client.messages.create(
                        model="claude-haiku-4-5-20251001",
                        max_tokens=600,
                        temperature=0,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return message.content[0].text.strip()
 
                _origin_prompt = (
                    f"You are a Pokemon encyclopaedia. In 3 to 4 concise sentences, "
                    f"describe the real-world design inspiration and origin of {_name_clean}. "
                    f"Cover the animals, mythology, objects, or cultural references its design "
                    f"draws from. Be factual and specific. Do not use bullet points."
                )
                _faq_prompt = (
                    f"You are a Pokemon encyclopaedia. Write exactly 4 frequently asked "
                    f"questions and answers about {_name_clean}. "
                    f"Format each strictly as: Q: question then A: answer on the next line. "
                    f"Focus on lore, competitive use, evolution, or notable trivia. "
                    f"Keep answers to 1 to 2 sentences each."
                )
 
                try:
                    st.session_state[_origin_key] = _call_claude(_origin_prompt)
                    st.session_state[_faq_key]    = _call_claude(_faq_prompt)
                except Exception as e:
                    st.error(f"API error: {e}")
                    st.session_state[_origin_key] = None
                    st.session_state[_faq_key]    = None
 
        _origin_text = st.session_state.get(_origin_key)
        _faq_text    = st.session_state.get(_faq_key)
 
        ai_left, ai_right = st.columns(2)
 
        with ai_left:
            st.subheader("Design Origin")
            if _origin_text:
                st.write(_origin_text)
            else:
                st.caption("Could not retrieve design origin.")
 
        with ai_right:
            st.subheader("FAQ")
            if _faq_text:
                for line in _faq_text.splitlines():
                    line = line.strip()
                    if not line:
                        st.markdown("")
                    elif line.startswith("Q:"):
                        st.markdown(f"**{line}**")
                    elif line.startswith("A:"):
                        st.markdown(line[2:].strip())
            else:
                st.caption("Could not retrieve FAQ.")