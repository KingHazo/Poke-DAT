import pandas as pd
import os
import re

# --- CONFIGURATION ---
CSV_FILE = 'pokemon_with_local_sprites.csv'
SPRITE_DIR = 'sprites'
# Set this to True to actually rename files. Set to False to just "test" it first.
ACTUAL_RENAME = True 

def clean_slug(text):
    """Matches the slugging logic used for the filenames."""
    if pd.isna(text): return ""
    # Standardize names: lowercase, remove special chars, convert spaces to hyphens
    text = text.lower()
    text = text.replace('♀', 'f').replace('♂', 'm')
    # Remove characters that don't usually appear in the filenames
    text = re.sub(r"[.'\"()]+", "", text)
    # Convert spaces and underscores to hyphens
    text = re.sub(r"[ _]+", "-", text)
    return text

def build_id_map(csv_path):
    """Creates a mapping of {name-slug: id}."""
    df = pd.read_csv(csv_path)
    id_map = {}
    
    for _, row in df.iterrows():
        dex_id = str(row['#'])
        full_name = row['Name']
        
        # We generate a slug for both the base name and the form name
        # e.g., "Charizard\nMega Charizard X" -> "charizard-megax"
        parts = full_name.split('\n')
        if len(parts) > 1:
            # It's a form. Use the logic used in previous steps to find the suffix
            base = clean_slug(parts[0])
            form = parts[1].lower()
            suffix = ""
            if "mega x" in form: suffix = "megax"
            elif "mega y" in form: suffix = "megay"
            elif "mega" in form: suffix = "mega"
            elif "alolan" in form: suffix = "alola"
            elif "galarian" in form: suffix = "galar"
            elif "hisuian" in form: suffix = "hisui"
            elif "paldean" in form: suffix = "paldea"
            elif "origin" in form: suffix = "origin"
            else: suffix = clean_slug(form.replace(parts[0].lower(), "").strip())
            
            slug = f"{base}-{suffix}" if suffix else base
        else:
            slug = clean_slug(parts[0])
        
        id_map[slug] = dex_id
    
    return id_map

def rename_sprites():
    if not os.path.exists(SPRITE_DIR):
        print(f"Error: Folder '{SPRITE_DIR}' not found.")
        return

    print("Building ID mapping from CSV...")
    id_map = build_id_map(CSV_FILE)
    
    print("Processing files...")
    files = [f for f in os.listdir(SPRITE_DIR) if f.startswith('imgi_') and f.endswith('.png')]
    
    success_count = 0
    fail_count = 0

    for filename in files:
        # 1. Extract the 'name' part (e.g., from imgi_15_aggron-mega.png -> aggron-mega)
        # We split by '_' and take everything after the second underscore
        parts = filename.split('_')
        if len(parts) < 3:
            continue
            
        name_part_with_ext = "_".join(parts[2:]) # Handles names that might have underscores
        name_part = name_part_with_ext.replace('.png', '')
        
        # 2. Look up the correct ID
        # Try exact match first, then try cleaning it
        correct_id = id_map.get(name_part)
        
        if not correct_id:
            # Try matching the base name if it's a gender variation like 'abomasnow-f'
            base_search = name_part.rsplit('-', 1)[0]
            correct_id = id_map.get(base_search)

        if correct_id:
            new_filename = f"{correct_id}-{name_part}.png"
            old_path = os.path.join(SPRITE_DIR, filename)
            new_path = os.path.join(SPRITE_DIR, new_filename)
            
            if ACTUAL_RENAME:
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_filename}")
            else:
                print(f"Would rename: {filename} -> {new_filename}")
            success_count += 1
        else:
            print(f"Could not find ID for: {filename}")
            fail_count += 1

    print(f"\nFinished Successfully renamed: {success_count}, Failed: {fail_count}")

if __name__ == "__main__":
    rename_sprites()