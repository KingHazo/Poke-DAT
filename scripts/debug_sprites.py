#Sprite Debugger Script

import streamlit as st
import pandas as pd
import os
from pathlib import Path

st.title("Sprite Path Debugger")

#Load  data
try:
    df = pd.read_csv('pokemon.csv')
    st.success(f"Loaded pokemon.csv with {len(df)} entries")
except Exception as e:
    st.error(f"Failed to load pokemon.csv: {e}")
    st.stop()

#Check if Local_Sprite column exists
if 'Local_Sprite' not in df.columns:
    st.error("No 'Local_Sprite' column found in pokemon.csv")
    st.write("Available columns:", df.columns.tolist())
    st.stop()

st.success("✓ Found 'Local_Sprite' column")

#Show sample paths
st.subheader("Sample Sprite Paths from CSV")
sample_paths = df['Local_Sprite'].dropna().head(10)
for idx, path in enumerate(sample_paths, 1):
    st.write(f"{idx}. `{path}`")

#Get current working directory
cwd = os.getcwd()
st.subheader("Current Working Directory")
st.code(cwd)

#Check if sprites directory exists
sprites_dir = os.path.join(cwd, 'sprites')
st.subheader("Sprite Directory Check")

if os.path.exists(sprites_dir):
    st.success(f"Sprites directory exists at: {sprites_dir}")
    
    #Count files
    sprite_files = [f for f in os.listdir(sprites_dir) if f.endswith('.png')]
    st.write(f"Found {len(sprite_files)} PNG files")
    
    #Show first 10 files
    st.write("Sample files:")
    for f in sprite_files[:10]:
        st.write(f"  - {f}")
else:
    st.error(f"Sprites directory not found at: {sprites_dir}")
    
    # Check alternative locations
    st.write("Checking alternative locations")
    alternatives = [
        './sprites',
        '../sprites'
    ]
    
    for alt in alternatives:
        alt_path = os.path.join(cwd, alt)
        if os.path.exists(alt_path):
            st.success(f"Found sprites at: {alt_path}")

#Test actual sprite loading
st.subheader("Sprite Loading Test")

#Select a Pokémon to test
test_pokemon = st.selectbox("Select a Pokémon to test:", df['Name'].head(20))

#Get the sprite path
sprite_row = df[df['Name'] == test_pokemon].iloc[0]
sprite_path = sprite_row['Local_Sprite']

st.write(f"Sprite path from CSV: `{sprite_path}`")

# ry different path variations
st.write("Testing different path formats:")

path_variations = [
    sprite_path,                          # As is from CSV
    f"./{sprite_path}",                   # With ./
    os.path.join(cwd, sprite_path),       # Absolute path
    sprite_path.replace('sprites/', ''),  # Just filename
    f"sprites/{sprite_path.split('/')[-1]}"  # Reconstruct
]

for i, test_path in enumerate(path_variations, 1):
    st.write(f"\n**Test {i}:** `{test_path}`")
    
    if os.path.exists(test_path):
        st.success("✓ File exists!")
        
        try:
            st.image(test_path, width=150, caption=f"Path {i} works")
        except Exception as e:
            st.error(f"File exists but failed to display: {e}")
    else:
        st.error(f"File not found at this path")
