import streamlit as st
import pandas as pd
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
# ============================================================================
# UI STARTS HERE
# ============================================================================

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
    "Machine Learning"
])

#MAIN TAB 1: RANKINGS
with main_tabs[0]:
    mode = st.radio("View Rankings By:", ["Type", "Specific Stat"], horizontal=True)
    
    if mode == "Type":
        #SUB-SECTION: Top 10 by Type 
        st.header("Top 10 Most Powerful Pokémon by Type")
        unique_types = sorted(df['Type_1'].unique())
        with st.popover(f"Filter by Type: {st.session_state.get('type_choice', 'Bug')}"):
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
            
    else:
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


#MAIN TAB 2: TRENDS
with main_tabs[1]:
    trend_mode = st.radio("View Trends By:", ["Type Distribution", "Average Power Level", "Base Stat Averages by Generation"], horizontal=True)
    
    if trend_mode == "Type Distribution":
        #SUB-SECTION: Distribution
        st.header("Pokémon Type Distribution")
        dist_choice = st.radio("Show Distribution for:", ["Primary Typing (Type 1)", "Secondary Typing (Type 2)"], horizontal=True)

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
    relationship_mode = st.radio("View Relationships By:", 
                                 ["Stat Correlation", "Pokemon Comparison Chart"], 
                                 horizontal=True)

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

        with selection_col1:
            p1 = st.selectbox("Select Pokémon 1", df['Name'].unique(), index=0, key="radar_p1")

        with selection_col2:
            p2 = st.selectbox("Select Pokémon 2", df['Name'].unique(), index=1, key="radar_p2")

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
                margin=dict(l=80, r=80, t=20, b=20)
            )

            return fig

        #Left Sprite/Radar Chart/Right Sprite
        pad_left, chart_middle, pad_right = st.columns([1, 5, 1])

        with chart_middle:
            st.plotly_chart(create_radar(p1, p2), use_container_width=True)

        #Sprites in a balanced two-column strip below the chart.
        #Each column gets exactly half the page, so both sprites scale identically
        sprite_col1, sprite_col2 = st.columns(2)

        sprite1_path = get_sprite_path(p1, df)
        sprite2_path = get_sprite_path(p2, df)
        type1_sprites = get_type_sprite_paths(p1, df)
        type2_sprites = get_type_sprite_paths(p2, df)


        def render_pokemon_panel(name, sprite_path, type_sprites):
            st.markdown(f"<h3 style='text-align: center;'>{name}</h3>", unsafe_allow_html=True)

            #Pokémon sprite so it doesn't stretch on wide screens
            _, img_area, _ = st.columns([0.5, 3, 0.5])
            with img_area:
                if sprite_path:
                    st.image(sprite_path, use_container_width=True)
                else:
                    st.caption("Sprite not available")

            #Type badge(s) centred beneath the sprite.
            #For a single type: one badge centred. For dual types: two badges side by side.
            if type_sprites:
                num_types = len(type_sprites)
                #Build a centred layout: narrow padding columns either side of the badges
                if num_types == 1:
                    _, badge_col, _ = st.columns([1.5, 1, 1.5])
                    badge_cols = [badge_col]
                else:
                    _, b1, b2, _ = st.columns([1, 1, 1, 1])
                    badge_cols = [b1, b2]

                for (type_name, type_path), col in zip(type_sprites, badge_cols):
                    with col:
                        st.image(type_path)
        with sprite_col1:
            render_pokemon_panel(p1, sprite1_path, type1_sprites)

        with sprite_col2:
            render_pokemon_panel(p2, sprite2_path, type2_sprites)

#MAIN TAB 4: MACHINE LEARNING
with main_tabs[3]:
    ml_mode = st.radio("View Machine Learning Method By:", ["K-Means", "DBSCAN"], horizontal=True)
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
    