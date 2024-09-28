from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import pandas as pd

# Function to get the page content
def get_page_content(url, headers):
    req = Request(url, headers=headers)
    return urlopen(req)

# Function to determine if the page is a fighter's page
def is_fighter_page(soup, fighter_name):
    # print("Checking if it's a page")
    meta_description = soup.find('meta', attrs={'name': 'description'})
    if meta_description:
        content = meta_description.get('content', '')
        # Check if the fighter's name (with spaces) is in the meta description
        if fighter_name.replace('-', ' ') in content:
            return True
    return False

def extract_name_from_a(tag):
    return tag.get_text(strip=True) if tag else ''

# Extract stats from fighter page
def analyze_fighter_page(soup, fighter):
    fights = []
    table_body = soup.find('tbody')
    rows = table_body.find_all('tr')
    event = ''
    date = ''
    for i in range(0, len(rows) - 1):
        if i % 3 == 0:
            continue  # Skip the event header row

        if i % 3 == 1:
            fighter_data = rows[i].find_all('td')
            fighter_name_tag = rows[i].find('th', class_='oppcell').find('a')
            fighter_name = extract_name_from_a(fighter_name_tag)
            fighter_odds = [td.get_text(strip=True) for td in fighter_data][0]
            event = rows[i].find('td', class_='item-non-mobile').find('a').get_text()

        if i % 3 == 2:
            opponent_data = rows[i].find_all('td')
            opponent_name_tag = rows[i].find('th', class_='oppcell').find('a')
            opponent_name = extract_name_from_a(opponent_name_tag)
            opponent_odds = [td.get_text(strip=True) for td in opponent_data][0]
            date = rows[i].find('td', class_='item-non-mobile').get_text()

            if 'UFC' in event:
                fights.append({
                    'Fighter': fighter_name,
                    'Fighter Odds': fighter_odds,
                    'Opponent': opponent_name,
                    'Opponent Odds': opponent_odds,
                    'Event': event,
                    'Date': date
                })

    df = pd.DataFrame(fights)
    return df

# Get fighter url
def analyze_search_results_page(soup, fighter, i):
    results = soup.find_all('tr')
    for result in results:
        fighter_link = result.find('a')
        if fighter_link:
            result_name = fighter_link.text.strip()
            if result_name == fighter.replace('-', ' '):
            # if result_name.lower() == fighter.replace('-', ' ').lower():
                fighter_url = 'https://www.bestfightodds.com' + fighter_link['href']
                print(f"Found Fighter: {result_name} -> {fighter_url}")
                return fighter_url
    print(f"No exact match found for {fighter}")
    return None

# Main function for getting fighter odds and stats
def main(url, fighter, i):
    head = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.84 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
        'Referer': url,
        'Cookie': (
            '_ga_GN2KEKQRH3=GS1.1.1723222411.4.1.1723224333.0.0.0; '
            '_ga=GA1.1.1118362177.1722220398; '
            'nexus_cookie={"is_new_session":"false","user_id":"efb66c14-c890-42cc-a8df-1c5a1f704886","last_session_id":"7f5026e5-9419-4f50-b2c5-fd62601252b9","last_session_start":"1723224326465","referrer_og":"https://www.google.com/"}; '
            '_omappvp=xDQ3uo5ZWj8AI9nScwOwxZFIVUoG7BEb7LhKpIVoR1ksfkW5IhL6LTkEtyzHRxwfNaUjFpblvGsitKgF5Dw3I703UeBEvZex; '
            '__cf_bm=GnirSoLIhSzz_2PKEYXusEULWIwPvHv.0Jwnvr2G5F8-1723224192-1.0.1.1-Ge0e6cp50RwyrvkWfq6kDiT_DiYgIvyR0IJYpOHagrqyv9uMrTJRIu.IxQlqEYaSBZyYAvXcho5zUIy9XPgtpw'
        )
    }

    try:
        response = get_page_content(url, head)
        page_content = response.read()

        soup = BeautifulSoup(page_content, 'html.parser')

        if is_fighter_page(soup, fighter):
            print(f"Analyzing fighter page for {fighter}")
            return analyze_fighter_page(soup, fighter)
        else:
            if i == 0:
                fighter_url = analyze_search_results_page(soup, fighter, i)
                if fighter_url:
                    return main(fighter_url, fighter, i + 1)
            else:
                print(f"Already tried for {fighter}")

    except Exception as e:
        print(f"Failed to fetch results for {fighter}: {e}")

    return None

if __name__ == "__main__":
    # Read the CSV file
    df = pd.read_csv('fighter_names.csv')
    # df = pd.read_csv('missing_fighters_916.csv')
    
    df['Name'] = df['Name'].str.replace(' ', '-')
    # Ensure only one column is present or select the specific column
    column_name = df.columns[1]  # Assuming the first column is the one you want
    fighters = df[column_name].to_numpy()
    
    # fighters = fighters[4000:]

    all_fights = []
    for fighter in fighters:
        url = 'https://www.bestfightodds.com/search?query=' + fighter
        df = main(url, fighter, 0)
        if df is not None:
            all_fights.append(df)

    # Concatenate all the dataframes into one
    if all_fights:
        final_df = pd.concat(all_fights, ignore_index=True)
        final_df.to_csv('fights_odds_921.csv', index=False)
        # print(final_df)
    else:
        print("No dataframes to concatenate.")
