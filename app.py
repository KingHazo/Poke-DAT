import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
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
    df = pd.read_csv('pokemon.csv')
    
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
        'Balanced':         '⚖️',
    }

    # Convert raw averages to per-Pokémon stat ratios at the cluster level
    normed = cluster_summary.div(cluster_summary.sum(axis=1), axis=0)

    # Key step: deviation from the cross-cluster mean
    # Positive = this cluster emphasises this stat MORE than others do
    global_mean = normed.mean()
    dev = normed.sub(global_mean)

    labels = {}
    for cid, row in dev.iterrows():
        atk   = row.get('Attack',  0)
        spatk = row.get('Sp. Atk', 0)
        spd   = row.get('Speed',   0)
        dfn   = row.get('Defense', 0)
        spdef = row.get('Sp. Def', 0)
        hp    = row.get('HP',      0)

        n_bulk = sum(1 for s in ['Defense', 'Sp. Def', 'HP'] if s in selected_features)
        bulk = (dfn + spdef + hp) / max(n_bulk, 1)

        is_fast  = spd > 0
        is_bulky = bulk > 0 and spd <= 0

        if is_fast:
            label = 'Physical Sweeper' if atk >= spatk else 'Special Sweeper'
        elif is_bulky:
            label = 'Physical Wall' if dfn >= spdef else 'Special Wall'
        else:
            label = 'Bulky Attacker' if (atk > 0 or spatk > 0) else 'Balanced'

        labels[cid] = f"{ICONS.get(label, '❓')} {label}"

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
def get_sprite_path(pokemon_name, df):
    try:
        sprite_path = df[df['Name'] == pokemon_name]['Local_Sprite'].iloc[0]     
        return sprite_path if pd.notna(sprite_path) and sprite_path != '' else None
    except:
        return None
# ============================================================================
# UI STARTS HERE
# ============================================================================

# Title Section
st.title("The Poké-DAT")
st.markdown("### A Data Analysis & Visualization Tool")

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
    trend_mode = st.radio("View Trends By:", ["Type Distribution", "Average Power Level"], horizontal=True)
    
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
            
    else:
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

#MAIN TAB 3: RELATIONSHIPS
with main_tabs[2]:
    relationship_mode = st.radio("View Relationships By:", 
                                 ["Stat Correlation", "Pokemon Comparison Chart"], 
                                 horizontal=True)

    if relationship_mode == "Stat Correlation":
        st.header("Stat Correlations")
        left, mid, right = st.columns([2, 4, 2])
        with mid:
            corr = get_stat_correlation(df)
            fig_corr, ax_corr = plt.subplots()
            sns.heatmap(corr, annot=True, cmap='RdYlGn', ax=ax_corr, 
                       linewidths=0.5, linecolor='lightgrey')
            st.pyplot(fig_corr)
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

        with sprite_col1:
            st.markdown(f"<h3 style='text-align: center;'>{p1}</h3>", unsafe_allow_html=True)
            #Inner padding so sprite doesn't balloon on very wide screens
            _, img_area, _ = st.columns([0.5, 3, 0.5])
            with img_area:
                if sprite1_path:
                    st.image(sprite1_path, use_container_width=True)
                else:
                    st.caption("Sprite not available")

        with sprite_col2:
            st.markdown(f"<h3 style='text-align: center;'>{p2}</h3>", unsafe_allow_html=True)
            _, img_area, _ = st.columns([0.5, 3, 0.5])
            with img_area:
                if sprite2_path:
                    st.image(sprite2_path, use_container_width=True)
                else:
                    st.caption("Sprite not available")

#MAIN TAB 4: MACHINE LEARNING
with main_tabs[3]:
    st.header("Pokémon Archetype Clustering")
    st.write("Use Machine Learning to group Pokémon based on Competitive Archetypes.")
    
    @st.fragment
    def clustering_analysis():
        
        selected_features = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
        k_val = 6
        
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

        with col_right:
            st.write("**Archetype Average Stats**")
            display_summary = cluster_summary.copy()
            display_summary.index = [archetype_labels[i] for i in display_summary.index]
            st.dataframe(display_summary)
    
    clustering_analysis()