import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

#Page Config
st.set_page_config(page_title="Poke-DAT", layout="wide")

#The Pokedex Theme
def get_pokedex_colors(n):
    # Defining a gradient based on classic Pokedex colors
    colors_list = ['#EF5350', '#EC407A', '#AB47BC', '#7E57C2', '#5C6BC0', '#42A5F5']
    cmap = LinearSegmentedColormap.from_list('pokedex', colors_list)
    return [cmap(i/n) for i in range(n)]

#Data Loading
@st.cache_data
def load_data():
    df = pd.read_csv('pokemon.csv')
    
    #Pre-processing
    df['Type_1'] = df['Type'].apply(lambda x: str(x).split('\n')[0].strip() if pd.notna(x) else 'Unknown')
    df['Type_2'] = df['Type'].apply(lambda x: str(x).split('\n')[1].strip() if pd.notna(x) and '\n' in str(x) else None)
    return df

df = load_data()

# --- Title Section ---
st.title("The Poké-DAT")
st.markdown("### A Data Analysis & Visualization Tool")

# Create the 4 tabs
tabs = st.tabs([
    "Top 10 by Type", 
    "Stat Leaderboard", 
    "Type Distribution", 
    "Average Power Level"
])

# --- TAB 1: Top 10 by Type ---
with tabs[0]:
    st.header("Top 10 Most Powerful Pokémon by Type")
    unique_types = sorted(df['Type_1'].unique())
    with st.popover(f"Filter by Type: {st.session_state.get('type_choice', 'Normal')}"):
            selected_type = st.radio(
                "Select a Pokémon Type:", 
                unique_types, 
                key="type_choice"
            )

    left, mid, right = st.columns([1, 4, 1])

    with mid:
        filtered_df = df[df['Type_1'] == selected_type].nlargest(10, 'Total')

        fig, ax = plt.subplots(figsize=(9, 5))
        colors = get_pokedex_colors(len(filtered_df))
        ax.barh(filtered_df['Name'].str.replace('\n', ' '), filtered_df['Total'], color=colors)

        for i, v in enumerate(filtered_df['Total']):
            ax.text(v + 2, i, f'{int(v)}', va='center', fontweight='bold', fontsize=8)

        ax.set_xlabel('Total Stats')
        ax.invert_yaxis()
        st.pyplot(fig)

#TAB 2: Specific Stats
with tabs[1]:
    st.header("Top 10 Pokémon by Specific Stat")
    stat_choice = st.segmented_control(
        "Select Stat", 
        ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed'],
        default='Total'
    )

    left, mid, right = st.columns([1, 4, 1])

    with mid:
        top_stat_df = df.nlargest(10, stat_choice)

        fig2, ax2 = plt.subplots(figsize=(9, 5))
        colors2 = get_pokedex_colors(10)
        ax2.barh(top_stat_df['Name'].str.replace('\n', ' '), top_stat_df[stat_choice], color=colors2)

        for i, v in enumerate(top_stat_df[stat_choice]):
            ax2.text(v + 2, i, f'{int(v)}', va='center', fontweight='bold', fontsize=8)

        ax2.invert_yaxis()
        st.pyplot(fig2)

#TAB 3: Distribution
with tabs[2]:
    st.header("Pokémon Type Distribution")
    dist_choice = st.radio("Show Distribution for:", ["Primary Typing (Type 1)", "Secondary Typing (Type 2)"])

    left, mid, right = st.columns([1, 5, 1])
    with mid:
        col_name = 'Type_1' if dist_choice == "Primary Typing (Type 1)" else 'Type_2'
        type_counts = df[col_name].value_counts().sort_values(ascending=False)

        fig3, ax3 = plt.subplots(figsize=(10, 4))
        type_counts.plot(kind='bar', color='#EF5350', ax=ax3)

        ax3.set_title(f"Distribution of {dist_choice}", fontsize=10, fontweight='bold')
        ax3.set_xlabel(dist_choice, fontsize=8)
        ax3.set_ylabel("Count", fontsize=8)

        for i, v in enumerate(type_counts.values):
            ax3.text(i, v, str(v), ha='center', va='bottom', fontweight='bold', fontsize=8)

        st.pyplot(fig3)

#TAB 4: Average Power Level
with tabs[3]:
    st.header("Average Power Level by Pokémon Type")

    left, mid, right = st.columns([2, 4, 2])
    with mid:
        avg_stats = df.groupby('Type_1')['Total'].mean().sort_values(ascending=False)
    
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
            ax4.text(v, i, f'{rounded_val}', va='center', fontweight='bold', fontsize=8)
        ax4.invert_yaxis()
        ax4.grid(axis='x', alpha=0.3, linestyle='--')

        st.pyplot(fig4)