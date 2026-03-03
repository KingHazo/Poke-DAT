import pandas as pd
import requests
import bs4
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

#Session
session = requests.Session()
retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))
session.mount('http://',  HTTPAdapter(max_retries=retries))
session.headers.update({'User-Agent': 'PokemonCSVEnricher/1.0 (educational project)'})

MAIN_PAGE_URL = 'https://bulbapedia.bulbagarden.net'
MEGA_PREFIXES = ('Mega ', 'Gigantamax ', 'Eternamax ', 'Primal ')

FIELDS = [
    'Height_m', 'Weight_kg', 'Gender_Male_Pct',
    'Catch_Rate', 'Egg_Cycles', 'Egg_Group',
    'Leveling_Rate', 'Base_Friendship',
    'Is_Legendary', 'Is_Mythical', 'Is_Ultra_Beast',
    'Abilities', 'Hidden_Ability',
]


#Thread-safe page cache
#Some Pokemon need the same Bulbapedia page.
#With 6 worker threads they can arrive simultaneously.
#
#Basically make a decision lock
#
#First thread:   creates a threading.Event, stores it, becomes the fetcher.
#Later threads:  find the Event already there, exit the lock, call event.wait().
#                   Physically blocked until the fetcher calls event.set().
#
#Two threads can never both become the fetcher for the same page.

_page_cache:  dict = {}   # page_name -> BeautifulSoup | None
_page_events: dict = {}   # page_name -> threading.Event
_cache_lock = threading.Lock()


def _fetch_soup(page_name: str):
    #Everything up to "i_am_fetcher = True/False" is atomic inside one lock.
    with _cache_lock:
        if page_name in _page_cache:
            return _page_cache[page_name]       #already fetched (hit or miss)

        if page_name in _page_events:
            event = _page_events[page_name]
            i_am_fetcher = False                #another thread is fetching
        else:
            event = threading.Event()
            _page_events[page_name] = event
            i_am_fetcher = True                 #we claimed this page
    #Lock released — branching is now safe and race-free.

    if not i_am_fetcher:
        event.wait(timeout=60)                  #block until fetcher signals done
        with _cache_lock:
            return _page_cache.get(page_name)

    #We are the fetcher — HTTP request with no locks held.
    url = f"{MAIN_PAGE_URL}/wiki/{page_name.replace(' ', '_')}_(Pok%C3%A9mon)"
    soup = None
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        soup = bs4.BeautifulSoup(resp.text, 'lxml')
    except Exception:
        pass

    with _cache_lock:
        _page_cache[page_name] = soup
    event.set()     #wake every thread that called event.wait()
    return soup


#Name parsing
def _parse_name(full_name: str):
    """
    Returns (page_name, form_name, is_base_form).

    CSV naming patterns:
      'Venusaur'                 -> page='Venusaur',  form='Venusaur',        base=True
      'Mega Venusaur'            -> page='Venusaur',  form='Mega Venusaur',   base=False
      'Meowth\nGalarian Meowth'  -> page='Meowth',   form='Galarian Meowth', base=False
    """
    parts = full_name.split('\n')
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip(), False

    name = parts[0].strip()
    for prefix in MEGA_PREFIXES:
        if name.startswith(prefix):
            page = name[len(prefix):]
            page = re.sub(r'\s+[XY]$', '', page)   # 'Mega Charizard X' -> 'Charizard'
            return page, name, False

    return name, name, True


#Utilities
def _cell_text(tag) -> str:
    return re.sub(r'\s+', ' ', tag.get_text(' ', strip=True)) if tag else ''


def _form_matches(attribution: str, form_name: str) -> bool:
    """
    Check whether form_name appears in attribution as a complete standalone
    token — NOT as the tail of a longer compound name like 'Mega Venusaur'.

    attribution = td_text with all ability link names stripped out, so every
    remaining Title-Case word is part of a form name. Rule: reject any match
    whose immediately preceding word starts with an uppercase letter.

    'Venusaur'       vs 'Venusaur'       -> True
    'Mega Venusaur'  vs 'Venusaur'       -> False  (preceded by 'Mega')
    'Alolan Rattata' vs 'Rattata'        -> False  (preceded by 'Alolan')
    'Alolan Rattata' vs 'Alolan Rattata' -> True
    '(Galarian Meowth)' vs 'Meowth'     -> False  (preceded by 'Galarian')
    """
    escaped = re.escape(form_name)
    for m in re.finditer(rf'(?<![A-Za-z]){escaped}(?![A-Za-z])', attribution):
        start = m.start()
        preceding = attribution[:start].rstrip()
        if preceding:
            last_word = [w for w in re.split(r'[\s()\[\],;:/]', preceding) if w]
            if last_word and last_word[-1][0].isupper():
                continue    # tail of a compound form name
        return True
    return False


# ── Abilities ──────────────────────────────────────────────────────────────────
def _parse_abilities(soup, form_name: str, is_base_form: bool, page_name: str):
    """
    Each ability <a> lives in a <td> whose text is:
      ABILITY_NAME(s)  FORM_CAPTION  ['Hidden Ability']

    e.g.  'Overgrow Venusaur'
          'Thick Fat Mega Venusaur'
          'Chlorophyll Hidden Ability'
          'Pickup or Tough Claws (Galarian Meowth)'
          'Gluttony Alolan Rattata'

    Algorithm:
      1. Strip all ability link texts from td_text -> attribution (form caption only).
      2. If attribution has any Title-Case word -> form-specific; only include if
         _form_matches(attribution, target_form).
      3. No Title-Case word remaining -> shared ability, include for every form.
      4. Check td text for 'Hidden Ability'.
    """
    target = form_name
    normal: list = []
    hidden: list = []

    for a in soup.find_all('a', title=lambda t: t and '(Ability)' in t):
        ab_name = a.get_text(strip=True)
        if ab_name == 'Cacophony':
            continue

        parent_td = a.find_parent('td')
        if not parent_td:
            continue

        td_text   = _cell_text(parent_td)
        is_hidden = (
            'Hidden Ability' in td_text or
            any('Hidden Ability' in s.get_text() for s in parent_td.find_all('small'))
        )

        # Build attribution: remove ability names and connector noise
        attribution = td_text
        for ab_a in parent_td.find_all('a', title=lambda t: t and '(Ability)' in t):
            attribution = attribution.replace(ab_a.get_text(strip=True), '')
        attribution = re.sub(
            r'\bor\b|Hidden Ability|Gen\s+\w+\+?|\bHP\b|[/()]', ' ', attribution
        ).strip()

        has_form_attr = bool(re.search(r'[A-Z][a-z]', attribution))

        if has_form_attr and not _form_matches(attribution, target):
            continue

        if is_hidden:
            hidden.append(ab_name)
        else:
            normal.append(ab_name)

    return (
        ', '.join(dict.fromkeys(normal)) or 'N/A',
        ', '.join(dict.fromkeys(hidden)) or 'None',
    )


#Height / Weight
def _parse_height_weight(soup, form_name: str, is_base_form: bool, page_name: str):
    h_anchor = soup.find('a', title='List of Pokémon by height')
    w_anchor = soup.find('a', title='Weight')
    height = _hw_from_cell(
        h_anchor.find_parent('td') if h_anchor else None,
        form_name, is_base_form, page_name, 'm')
    weight = _hw_from_cell(
        w_anchor.find_parent('td') if w_anchor else None,
        form_name, is_base_form, page_name, 'kg')
    return height, weight


def _hw_from_cell(td, form_name, is_base_form, page_name, unit) -> str:
    if td is None:
        return 'N/A'
    pairs: list = []
    cur = None
    for node in td.descendants:
        if isinstance(node, bs4.NavigableString):
            txt = node.strip()
            if txt and re.search(rf'[\d?].*{re.escape(unit)}', txt):
                cur = txt
        elif getattr(node, 'name', None) == 'small':
            label = node.get_text(strip=True)
            if cur is not None:
                pairs.append((cur, label))
                cur = None

    if not pairs:
        m = re.search(rf'([\d.?+]+\s*\+?\s*{re.escape(unit)})', td.get_text())
        return _metric_only(m.group(1), unit) if m else 'N/A'

    target = form_name if not is_base_form else page_name
    for value, label in pairs:
        if target.lower() in label.lower() or label.lower() in target.lower():
            return _metric_only(value, unit)
    if is_base_form and pairs:
        return _metric_only(pairs[0][0], unit)
    return 'N/A'


def _metric_only(text: str, unit: str) -> str:
    m = re.search(rf'([\d.?+]+)\s*\+?\s*{re.escape(unit)}', text)
    return m.group(1) if m else text.strip()


#Vitals helpers
def _vitals_value(soup, label_title: str):
    a = soup.find('a', title=label_title)
    if not a:
        return None
    td = a.find_parent('td')
    if not td:
        return None
    return _cell_text(td).replace(a.get_text(strip=True), '').strip(' :--\xa0') or None


def _parse_gender(soup) -> str:
    for tag in soup.find_all(['span', 'td', 'a']):
        txt = tag.get_text(' ', strip=True).replace('\xa0', ' ')
        m = re.search(r'([\d.]+)\s*%\s*male', txt)
        if m:
            return m.group(1)
    return 'N/A'


def _parse_classifications(soup):
    cats = ' '.join(
        a.get_text(strip=True)
        for a in soup.find_all('a', href=lambda h: h and 'Category:' in h)
    )
    return (
        1 if 'Legendary' in cats and 'Pokemon' in cats else 0,
        1 if 'Mythical'  in cats and 'Pokemon' in cats else 0,
        1 if 'Ultra Beasts' in cats else 0,
    )


#Main per-row function
def get_enhanced_data(full_name: str) -> dict:
    page_name, form_name, is_base_form = _parse_name(full_name)

    data = {f: 'N/A' for f in FIELDS}
    data.update({'Is_Legendary': 0, 'Is_Mythical': 0, 'Is_Ultra_Beast': 0})

    soup = _fetch_soup(page_name)
    if soup is None:
        return data

    h, w = _parse_height_weight(soup, form_name, is_base_form, page_name)
    data['Height_m']  = h
    data['Weight_kg'] = w

    cr = _vitals_value(soup, 'Catch rate')
    if cr:
        m = re.search(r'(\d+)', cr)
        data['Catch_Rate'] = m.group(1) if m else cr

    ec = _vitals_value(soup, 'Egg cycle')
    if ec:
        m = re.search(r'(\d+)', ec)
        data['Egg_Cycles'] = m.group(1) if m else ec

    eg_tags = soup.find_all('a', title=lambda t: t and '(Egg Group)' in t)
    groups  = [g.get_text(strip=True) for g in eg_tags
               if 'Group' not in g.get_text(strip=True)]
    data['Egg_Group'] = ', '.join(dict.fromkeys(groups)) if groups else 'Undiscovered'

    lr_tag = soup.find(
        lambda tag: tag.name in ('b', 'span', 'a') and 'Leveling rate' in tag.get_text())
    if lr_tag:
        td = lr_tag.find_parent('td')
        if td:
            txt = _cell_text(td).replace('Leveling rate', '').strip(' :--\xa0')
            data['Leveling_Rate'] = txt or 'N/A'

    bf_a = soup.find('a', title='List of Pokémon by base friendship')
    if bf_a:
        td = bf_a.find_parent('td')
        if td:
            txt = _cell_text(td).replace('Base friendship', '').strip(' :--\xa0')
            m = re.search(r'(\d+)', txt)
            data['Base_Friendship'] = m.group(1) if m else txt

    data['Gender_Male_Pct'] = _parse_gender(soup)

    abilities, hidden = _parse_abilities(soup, form_name, is_base_form, page_name)
    data['Abilities']      = abilities
    data['Hidden_Ability'] = hidden

    leg, myth, ub = _parse_classifications(soup)
    data['Is_Legendary']   = leg
    data['Is_Mythical']    = myth
    data['Is_Ultra_Beast'] = ub

    return data


def process_row(index, row):
    return index, get_enhanced_data(row['Name'])


if __name__ == '__main__':
    import sys
    input_csv  = sys.argv[1] if len(sys.argv) > 1 else 'pokemon.csv'
    output_csv = sys.argv[2] if len(sys.argv) > 2 else 'pokemon_copy.csv'

    df = pd.read_csv(input_csv)
    results = [None] * len(df)

    print(f"Scraping {len(df)} rows from Bulbapedia...")
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(process_row, i, row): i
            for i, row in df.iterrows()
        }
        for future in tqdm(as_completed(futures), total=len(futures), unit='mon'):
            idx, res = future.result()
            results[idx] = res

    enhanced_df = pd.DataFrame(results, columns=FIELDS)
    final_df    = pd.concat([df, enhanced_df], axis=1)
    final_df.to_csv(output_csv, index=False)
    print(f"Done — saved to {output_csv}")