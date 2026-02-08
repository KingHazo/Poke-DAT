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
    # Defining a gradient based on classic Pokedex colors
    colors_list = ['#EF5350', '#EC407A', '#AB47BC', '#7E57C2', '#5C6BC0', '#42A5F5']
    cmap = LinearSegmentedColormap.from_list('pokedex', colors_list)
    return [cmap(i/n) for i in range(n)]

#Lightgrey background for Matplotlib/Seaborn charts since white hurts my eyes
plt.rcParams.update({
    "figure.facecolor": "darkgrey",  #The area around the chart
    "axes.facecolor": "darkgrey",    #The area inside the chart axes
    "savefig.facecolor": "darkgrey"  #Ensures it stays grey if saved
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

# --- MAIN TAB 1: RANKINGS ---
with main_tabs[0]:
    # Secondary navigation for Rankings
    mode = st.radio("View Rankings By:", ["Type", "Specific Stat"], horizontal=True)
    
    if mode == "Type":
        # --- SUB-SECTION: Top 10 by Type ---
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
            
    else:
        # --- SUB-SECTION: Specific Stats ---
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

# --- MAIN TAB 2: TRENDS ---
with main_tabs[1]:
    # Secondary navigation for Trends
    trend_mode = st.radio("View Trends By:", ["Type Distribution", "Average Power Level"], horizontal=True)
    
    if trend_mode == "Type Distribution":
        # --- SUB-SECTION: Distribution ---
        st.header("Pokémon Type Distribution")
        dist_choice = st.radio("Show Distribution for:", ["Primary Typing (Type 1)", "Secondary Typing (Type 2)"], horizontal=True)

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
            
    else:
        # --- SUB-SECTION: Average Power Level ---
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
                ax4.text(v + 3, i, f'{rounded_val}', va='center', fontweight='bold', fontsize=8)
            ax4.invert_yaxis()
            ax4.grid(axis='x', alpha=0.3, linestyle='--')
            st.pyplot(fig4)

# --- MAIN TAB 3: RELATIONSHIPS ---
with main_tabs[2]:
    relationship_mode = st.radio("View Relationships By:", ["Stat Correlation", "Pokemon Comparison Chart"], horizontal=True)

    if relationship_mode == "Stat Correlation":
        # --- SUB-SECTION: Stat Correlation Heatmap ---
        st.header("Stat Correlations")
        left, mid, right = st.columns([2, 4, 2])
        with mid:
            # Select only numeric columns
            corr = df[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']].corr()
            fig_corr, ax_corr = plt.subplots()
            sns.heatmap(corr, annot=True, cmap='RdYlGn', ax=ax_corr, linewidths=0.5, linecolor='lightgrey')
            st.pyplot(fig_corr)
    else:
        # --- SUB-SECTION: Pokemon Pentagon Charts ---
        st.header("Stat Radar Comparison")
        st.write("Compare the stat 'shape' of two Pokémon.")

        col_a, col_b = st.columns(2)
        with col_a:
            p1 = st.selectbox("Select Pokémon 1", df['Name'].unique(), index=0)
        with col_b:
            p2 = st.selectbox("Select Pokémon 2", df['Name'].unique(), index=0) # Default to 000 Bulbasaur

        def create_radar(name1, name2):
            categories = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
            fig = go.Figure()

            for name, color in [(name1, '#EF5350'), (name2, '#42A5F5')]:
                stats = df[df['Name'] == name][categories].values.flatten().tolist()
                stats.append(stats[0]) # Close the loop
                fig.add_trace(go.Scatterpolar(
                    r=stats, theta=categories + [categories[0]],
                    fill='toself', name=name, line_color=color
                ))
            fig.update_layout(
                polar=dict(
                    bgcolor="darkgrey", # Sets the circle background to grey
                    radialaxis=dict(
                        visible=True, 
                        range=[0, 200],
                    ),
                ),
                showlegend=True
            )

            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 200])), showlegend=True)
            return fig

        st.plotly_chart(create_radar(p1, p2), use_container_width=True)
# MAIN TAB 4: MACHINE LEARNING
with main_tabs[3]:
    st.header("Pokémon Archetype Clustering")
    st.write("Use Machine Learning to group Pokémon based on custom stat combinations.")
    
    # 1. Feature Selection
    all_stats = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
    
    col_input1, col_input2 = st.columns(2)
    with col_input1:
        selected_features = st.multiselect(
            "Select stats for the AI to analyze:",
            options=all_stats,
            default=all_stats # Defaults to using everything
        )
    with col_input2:
        k_val = st.slider("Number of Archetypes (Clusters):", 4, 6, 5)

    if len(selected_features) < 2:
        st.warning("Please select at least two stats to perform clustering.")
    else:
        # 2. K-Means Logic
        scaler = StandardScaler()
        scaled_stats = scaler.fit_transform(df[selected_features])
        
        kmeans = KMeans(n_clusters=k_val, random_state=42)
        df['Cluster'] = kmeans.fit_predict(scaled_stats)
        
        # 3. Plotting Controls
        st.markdown("---")
        st.subheader("Visualize the Results")
        col_plot1, col_plot2 = st.columns(2)
        with col_plot1:
            x_axis = st.selectbox("X-Axis Stat:", selected_features, index=0)
        with col_plot2:
            y_axis = st.selectbox("Y-Axis Stat:", selected_features, index=min(1, len(selected_features)-1))

        col_left, col_right = st.columns([3, 2])
        with col_left:          
            # Create interactive scatter plot
            fig_km = px.scatter(
                df, 
                x=x_axis, 
                y=y_axis, 
                color=df['Cluster'].astype(str), # Convert to string for discrete colors
                hover_name='Name',               # This shows the Pokémon name on hover
                title=f"Clusters View: {x_axis} vs {y_axis}",
                labels={'color': 'Cluster'},
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            
            fig_km.update_traces(marker=dict(size=10, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))
            
            # Display
            st.plotly_chart(fig_km, use_container_width=True)
            
        with col_right:
            st.write("**Archetype Average Stats**")
            # Show the average of the SELECTED stats for each cluster
            cluster_summary = df.groupby('Cluster')[selected_features].mean().round(0)
            st.dataframe(cluster_summary)

            # Fun Insight: Logic to identify the "Strongest" cluster
            top_cluster = cluster_summary.sum(axis=1).idxmax()
            st.info(f"Cluster {top_cluster} appears to be the most powerful group based on the selected stats.")