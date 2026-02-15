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
    """Perform K-Means clustering on selected features"""
    scaler = StandardScaler()
    scaled_stats = scaler.fit_transform(df[selected_features])
    
    kmeans = KMeans(n_clusters=k_val, random_state=42)
    clusters = kmeans.fit_predict(scaled_stats)
    
    # Return a new dataframe with cluster assignments
    df_clustered = df.copy()
    df_clustered['Cluster'] = clusters
    
    return df_clustered

@st.cache_data
def get_cluster_summary(df_clustered, selected_features):
    """Calculate cluster summary statistics"""
    return df_clustered.groupby('Cluster')[selected_features].mean().round(0)

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
        
        #Create two-column layout
        chart_col, sprites_col = st.columns([3.5, 1.5])
        
        with chart_col:
            #Dynamic height based on number of items
            num_items = len(filtered_df)
            fig_height = num_items * 0.4
            
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
        
        with sprites_col:
            st.markdown("### Pokémon")
            
            for rank, (idx, row) in enumerate(filtered_df.iterrows(), 1):
                sprite_path = get_sprite_path(row['Name'], df)

                cols = st.columns([2, 5, 19])
                
                with cols[0]:
                    st.markdown(f"**#{rank}**")
                
                with cols[1]:
                    if sprite_path:
                        st.image(sprite_path, width=40)
                    else:
                        st.write("Sprite Not Available")
                with cols[2]:
                    st.caption(f"**{row['Name'].replace('\n', ' ')}** • {int(row['Total'])}")
            
    else:
        #SUB-SECTION: Specific Stats
        st.header("Top 10 Pokémon by Specific Stat")
        stat_choice = st.segmented_control(
            "Select Stat", 
            ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed'],
            default='Total'
        )

        top_stat_df = get_top_pokemon_by_stat(df, stat_choice, 10)
        
        chart_col, sprites_col = st.columns([3.5, 1.5])
        
        with chart_col:
            num_items = len(top_stat_df)
            fig_height = num_items * 0.4
            
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
        
        with sprites_col:
            st.markdown(f"### Pokémon")
            
            for rank, (idx, row) in enumerate(top_stat_df.iterrows(), 1):
                sprite_path = get_sprite_path(row['Name'], df)
                
                cols = st.columns([2, 5, 19])
                
                with cols[0]:
                    st.markdown(f"**#{rank}**")
                
                with cols[1]:
                    if sprite_path:
                        st.image(sprite_path, width=40)
                    else:
                        st.write("Sprite Not Available")
                with cols[2]:
                    st.caption(f"**{row['Name'].replace('\n', ' ')}** • {int(row[stat_choice])}")


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
                    bgcolor="darkgrey",
                    radialaxis=dict(visible=True, range=[0, 255]),
                ),
                showlegend=True,
                height=500,
                margin=dict(l=80, r=80, t=20, b=20)
            )

            return fig

        #Left Sprite/Radar Chart/Right Sprite
        sprite_left, chart_middle, sprite_right = st.columns([1, 3, 1])

        #Left Pokémon Sprite
        with sprite_left:
            #Center the Pokémon name
            st.markdown(f"<h3 style='text-align: center;'>{p1}</h3>", unsafe_allow_html=True)

            #Get and display sprite
            sprite1_path = get_sprite_path(p1, df)

            if sprite1_path:
                st.image(sprite1_path, use_container_width=True)
            else:
                #Fallback
                st.markdown("<div style='text-align: center; font-size: 60px;'>🎮</div>", unsafe_allow_html=True)
                st.caption("Sprite not available", unsafe_allow_html=True)

        #Radar Chart
        with chart_middle:
            st.plotly_chart(create_radar(p1, p2), use_container_width=True)

        #Right Pokémon Sprite
        with sprite_right:
            #Center the Pokémon name
            st.markdown(f"<h3 style='text-align: center;'>{p2}</h3>", unsafe_allow_html=True)

            #Get and display sprite
            sprite2_path = get_sprite_path(p2, df)

            if sprite2_path:
                st.image(sprite2_path, use_container_width=True)
            else:
                # Fallback
                st.markdown("<div style='text-align: center; font-size: 60px;'>🎮</div>", unsafe_allow_html=True)
                st.caption("Sprite not available", unsafe_allow_html=True)

#MAIN TAB 4: MACHINE LEARNING
with main_tabs[3]:
    st.header("Pokémon Archetype Clustering")
    st.write("Use Machine Learning to group Pokémon based on custom stat combinations.")
    
    @st.fragment
    def clustering_analysis():
        """Interactive ML clustering"""
        
        col_input1, col_input2 = st.columns(2)
        with col_input1:
            selected_features = st.multiselect(
                "Select stats for the AI to analyze:",
                options=st.session_state.ml_stats,  #From session state
                default=st.session_state.ml_stats,
                key="ml_features"
            )
        with col_input2:
            k_val = st.slider("Number of Archetypes (Clusters):", 4, 6, 5, key="ml_clusters")

        if len(selected_features) < 2:
            st.warning("Please select at least two stats to perform clustering.")
            return  #Early exit
        
        #Clustering
        df_clustered = perform_clustering(df, selected_features, k_val)
        
        st.markdown("---")
        st.subheader("Visualize the Results")
        
        col_plot1, col_plot2 = st.columns(2)
        with col_plot1:
            x_axis = st.selectbox("X-Axis Stat:", selected_features, index=0, key="ml_x")
        with col_plot2:
            y_axis = st.selectbox("Y-Axis Stat:", selected_features, 
                                 index=min(1, len(selected_features)-1), key="ml_y")

        col_left, col_right = st.columns([3, 2])
        with col_left:          
            fig_km = px.scatter(
                df_clustered, 
                x=x_axis, 
                y=y_axis, 
                color=df_clustered['Cluster'].astype(str),
                hover_name='Name',
                title=f"Clusters View: {x_axis} vs {y_axis}",
                labels={'color': 'Cluster'},
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            
            fig_km.update_traces(marker=dict(size=10, opacity=0.7, 
                                            line=dict(width=1, color='DarkSlateGrey')))
            st.plotly_chart(fig_km, use_container_width=True)
            
        with col_right:
            st.write("**Archetype Average Stats**")
            cluster_summary = get_cluster_summary(df_clustered, selected_features)
            st.dataframe(cluster_summary)

            top_cluster = cluster_summary.sum(axis=1).idxmax()
            st.info(f"Cluster {top_cluster} appears to be the most powerful group.")
    
    clustering_analysis()