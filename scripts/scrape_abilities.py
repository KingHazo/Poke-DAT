import requests
from bs4 import BeautifulSoup
import pandas as pd
HEADERS = {'User-Agent': 'SmogonSetsScraper/1.0 (educational project)'}

def scrape_pokemon_abilities():
    url = "https://bulbapedia.bulbagarden.net/wiki/Ability"

    print("Connecting to Bulbapedia...")
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    
    #Find all tables and look for the one with the right "signature"
    all_tables = soup.find_all('table')
    target_table = None

    print(f"Scanning {len(all_tables)} tables for the Ability list...")

    for table in all_tables:
        #Small layout tables won't have this many rows.
        rows = table.find_all('tr')
        if len(rows) < 100:
            continue
            
        #Check the first two rows for our keywords
        header_text = table.get_text().lower()
        if "ability" in header_text and "description" in header_text:
            target_table = table
            break

    if not target_table:
        print("Error: Could not find a table containing 'Ability' and 'Description' with 100+ rows.")
        return

    ability_data = []
    #Re-scan the rows of our confirmed target table
    rows = target_table.find_all('tr')
    
    for row in rows:
        cols = row.find_all('td')
        
        #Bulbapedia's list table structure:
        #[0] Index (#), [1] Ability Name, [2] Description, [3] Generation
        if len(cols) >= 3:
            name = cols[1].text.strip()
            description = cols[2].text.strip()
            
            #Basic validation to ensure we're not grabbing a mid-table header row
            if name and name != "Ability":
                ability_data.append({
                    'Name': name,
                    'Description': description
                })

    if ability_data:
        df = pd.DataFrame(ability_data)
        #Final cleanup: Remove any lingering duplicate header rows or empty entries
        df = df.drop_duplicates().reset_index(drop=True)
        
        df.to_csv('pokemon_abilities.csv', index=False, encoding='utf-8')
        print(f"Success! Exported {len(df)} abilities to 'pokemon_abilities.csv'.")
    else:
        print("Found the table, but failed to extract data rows.")

if __name__ == "__main__":
    scrape_pokemon_abilities()