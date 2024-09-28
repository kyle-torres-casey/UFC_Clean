import pandas as pd
import numpy as np

def get_year(dob):
    if pd.isna(dob):
        return "unk"
    return dob.split(", ")[-1]

# This is tricky
def fix_duplicates(fighters, bouts_clean):
    # check if there are fighters with the same name
    print(fighters[fighters.duplicated(subset="Name", keep=False)])
    print("")
    name_col_index = fighters.columns.get_loc('Name')

    fighters.drop(3334, inplace=True)
    fighters.iloc[3317, name_col_index] = "Bruno Silva 185"
    fighters.drop(2660, inplace=True)
    fighters.drop(2282, inplace=True)
    fighters.drop(1543, inplace=True)
    fighters.drop(1491, inplace=True)
    fighters.drop(399, inplace=True)

    print(fighters[fighters.duplicated(subset="Name", keep=False)])

    # Add new names to bouts dataframe
    for col in ["Fighter 1", "Fighter 2"]:
        bouts_clean.loc[(bouts_clean[col] == "Bruno Silva") &
        (bouts_clean["Weight class"] == "Middleweight"), col] = "Bruno Silva 185"

    return fighters, bouts_clean

def drop_edit_col_names(fighters):
    # Create Name column
    fighters['Name'] = fighters['First'].fillna('') + ' ' + fighters['Last'].fillna('')
    # Strip any leading or trailing whitespace (in case both First and Last are NaN)
    fighters['Name'] = fighters['Name'].str.strip()

    fighters.drop("Nickname", axis=1, inplace=True)
    fighters.drop("First", axis=1, inplace=True)
    fighters.drop("Last", axis=1, inplace=True)
    fighters.drop("SLpM", axis=1, inplace=True)
    fighters.drop("Str. Acc.", axis=1, inplace=True)
    fighters.drop("SApM", axis=1, inplace=True)
    fighters.drop("Str. Def", axis=1, inplace=True)
    fighters.drop("TD Avg.", axis=1, inplace=True)
    fighters.drop("TD Acc.", axis=1, inplace=True)
    fighters.drop("TD Def.", axis=1, inplace=True)
    fighters.drop("Sub. Avg.", axis=1, inplace=True)
    fighters.drop("Belt", axis=1, inplace=True)
    fighters.drop("D", axis=1, inplace=True)
    fighters.drop("Unnamed: 15", axis=1, inplace=True)

    fighters.rename(columns={
        'W': 'Career W',
        'L': 'Career L',
        # Add more columns if needed
    }, inplace=True)

    # use born year only
    fighters["born_year"] = fighters["DOB"].map(lambda dob: get_year(dob))

    return fighters

# Function to convert height to inches
def height_to_inches(height):
    try:
        # Split the height into feet and inches
        if pd.isna(height) or '--' in height:
            return None
        parts = height.split("' ")
        feet = int(parts[0])
        inches = int(parts[1].replace('"', ''))

        # Convert to total inches
        total_inches = feet * 12 + inches
        return total_inches
    except (ValueError, IndexError):
        return None
    
def clean_fighter_stats(fighters):
    more_fighter_stats = fighters
    # Apply the function to the 'Ht' column
    more_fighter_stats['Ht'] = more_fighter_stats['Ht'].apply(height_to_inches)
    # Assuming 'Reach' is the name of your column
    more_fighter_stats['Reach'] = more_fighter_stats['Reach'].str.replace('"', '', regex=False)
    more_fighter_stats.drop("Wt", axis=1, inplace=True)
    more_fighter_stats.drop("DOB", axis=1, inplace=True)
    return more_fighter_stats

def create_fighter_dictionaries(bouts_clean, unique_fighters, more_fighter_stats):
    # Create Dictionary of Dataframes for each fighter's career in UFC
    fighter_dfs = {}
    for fighter in unique_fighters:
        # Filter rows where the fighter appears as either 'Fighter 1' or 'Fighter 2'
        fighter_df = bouts_clean[(bouts_clean['Fighter 1'] == fighter) | (bouts_clean['Fighter 2'] == fighter)].copy()

        # Remove rows where 'W/L 1' is 'NC' and 'draw'
        fighter_df = fighter_df[(fighter_df['W/L 1'] != 'nc') & (fighter_df['W/L 1'] != 'draw')]

        # Identify the rows where the fighter is 'Fighter 2'
        is_fighter2 = fighter_df['Fighter 2'] == fighter

        # Swap columns for rows where the fighter is 'Fighter 2'
        columns_to_swap = ['W/L', 'Kd', 'Str', 'Td', 'Sub', 'KD', 'Sig. str.', 'Sig. str. %', 
                           'Total str.', 'Td %', 'Sub. att', 'Rev.', 'Ctrl']

        for col in columns_to_swap:
            fighter_df.loc[is_fighter2, [f'{col} 1', f'{col} 2']] = fighter_df.loc[is_fighter2, [f'{col} 2', f'{col} 1']].values

        # Also swap 'Fighter 1' and 'Fighter 2' columns
        fighter_df.loc[is_fighter2, ['Fighter 1', 'Fighter 2']] = fighter_df.loc[is_fighter2, ['Fighter 2', 'Fighter 1']].values

        # Convert 'Date' column to datetime
        fighter_df['Date'] = pd.to_datetime(fighter_df['Date'], errors='coerce')

        # Sort the DataFrame by 'Date' column in ascending order
        fighter_df = fighter_df.sort_values(by='Date', ascending=True)

        # Reset the index to reflect the new order
        fighter_df = fighter_df.reset_index(drop=True)

        if fighter_df.empty:
            print(f"Deleting empty DataFrame for fighter: {fighter}")
        else:
            # Add additional columns from more_fighter_stats where 'Name' matches 'fighter'
            additional_stats = more_fighter_stats[more_fighter_stats['Name'] == fighter].drop(columns=['Name'])

            if not additional_stats.empty:
                for col in additional_stats.columns:
                    fighter_df[col] = additional_stats.iloc[0][col]
            else:
                print(f"No additional stats found for fighter: {fighter}")

            # Store the updated DataFrame in the dictionary
            fighter_dfs[fighter] = fighter_df

        
        if fighter == 'Gilbert Burns':
            print(fighter_df)
            fighter_df.to_csv("Data/burns.csv")
        
    
    return fighter_dfs

def initialize_columns(fighter_df):
    columns = {
        'W': 0.0, 'L': 0.0, 'Num Fights': 0.0, 'W Perc': 0.0,
        'Sig Strikes Avg': 0.0, 'Sig Str %': 0.0, 'Sig Strikes Opp Avg': 0.0, 'Sig Str % Opp': 0.0,
        'Strikes Avg': 0.0, 'Str %': 0.0, 'Strikes Opp Avg': 0.0, 'Str % Opp': 0.0,
        'TD Avg': 0.0, 'TD %': 0.0, 'TD Opp Avg': 0.0, 'TD % Opp': 0.0,
        'KD Avg': 0.0, 'KD Opp Avg': 0.0, 'DEC Avg': 0.0, 'KO Avg': 0.0, 'SUB Avg': 0.0,
        'DEC Opp Avg': 0.0, 'KO Opp Avg': 0.0, 'SUB Opp Avg': 0.0,
        'CTRL Avg': 0.0, 'CTRL Opp Avg': 0.0, 'Time Avg': 0.0, 'Streak': 0.0,
        'Ht Diff': 0.0, 'Reach Diff': 0.0, 'Age': 0.0
        # 'Career W': fighter_df['Career W'].values[len(fighter_df) -1 ], 
        # 'Career L': fighter_df['Career L'].values[len(fighter_df) -1 ]
        # 'Career Fights': 0.0, 'Career W Perc': 0.0
    }
    for col, value in columns.items():
        fighter_df.loc[:, col] = value

    return fighter_df

def initialize_running_stats(fighter_df):
    running_stats = {
        "wins": 0.0,  
        "losses": 0.0,
        "fights": 0.0,
        "w_perc": 0.0,

        # Sig Strikes
        "running_sig_strikes_1": 0.0,
        "running_attempted_strikes_1": 0.0,
        "running_sig_strikes_2": 0.0,
        "running_attempted_strikes_2": 0.0,

        # Total Strikes
        "running_strikes_1": 0.0,
        "running_attempted_strikes_1": 0.0,
        "running_strikes_2": 0.0,
        "running_attempted_strikes_2": 0.0,

        # Total TD (Takedowns)
        "running_td_1": 0.0,
        "running_attempted_td_1": 0.0,
        "running_td_2": 0.0,
        "running_attempted_td_2": 0.0,

        # Knockdowns (KD)
        "running_kd_1": 0.0,
        "running_kd_2": 0.0,

        # Method: Decision (DEC), KO, Submission (SUB)
        "running_dec": 0.0,
        "running_ko": 0.0,
        "running_sub": 0.0,
        "running_dec_opp": 0.0,
        "running_ko_opp": 0.0,
        "running_sub_opp": 0.0,

        # Control Time
        "running_ctrl_1": 0.0,
        "running_ctrl_2": 0.0,

        # Fight Time
        "running_time": 0.0,

        # Streak
        "streak": 0.0,

        #Career wins and losses
        "career_w": fighter_df['Career W'].values[len(fighter_df) - 1],
        "career_l": fighter_df['Career L'].values[len(fighter_df) - 1]
    }
        
    return running_stats

def clean_columns(fighter_df):
    # Remove the specified columns
    fighter_df = fighter_df.drop(columns=[
        'Sig. str. 1', 'Sig. str. 2', 'Sig. str. % 1', 'Sig. str. % 2',
        'Total str. 1', 'Total str. 2', 'W/L 2', 'Rev. 1', 'Rev. 2', 'Sub. att 1',
        'Sub. att 2', 'Sub 1', 'Sub 2', 'Str 1', 'Str 2', 'Td % 1', 'Td % 2',
        'Td 1', 'Td 2', 'Kd 1', 'Kd 2', 'Method', 'Round', 'Ctrl 1', 'Ctrl 2', 'Time',
        'KD 1', 'KD 2', 'Ht', 'Reach'
    ])
    return fighter_df

def update_physical(i, fighter_df, more_fighter_stats, fighter):
    # Get the opponent's stats
    opponent = fighter_df.iloc[i]['Fighter 2']
    opponent_row = more_fighter_stats.loc[more_fighter_stats['Name'] == opponent]
    if not opponent_row.empty:
        # Get opponent's height and fighter's height as scalar values
        opponent_ht = opponent_row['Ht'].values[0] if not opponent_row['Ht'].isna().all() else np.nan
        ht = fighter_df.iloc[i]['Ht'] if not pd.isna(fighter_df.iloc[i]['Ht']) else np.nan
        # Check for NaN using np.isnan()
        if not np.isnan(opponent_ht) and not np.isnan(ht):
            fighter_df.loc[fighter_df.index[i], 'Ht Diff'] = ht - opponent_ht
        else:
            print(f"Empty value for height for fighter: {fighter}")

        #Reach
        # Get opponent's reach and fighter's reach as scalar values
        opponent_reach = opponent_row['Reach'].values[0] if not opponent_row['Reach'].isna().all() else '--'
        reach = fighter_df.iloc[i]['Reach'] if not pd.isna(fighter_df.iloc[i]['Reach']) else '--'
        # Compare the values
        if opponent_reach != '--' and reach != '--':
            fighter_df.loc[fighter_df.index[i], 'Reach Diff'] = float(reach) - float(opponent_reach)
        else:
            fighter_df.loc[fighter_df.index[i], 'Reach Diff'] = 0
    return fighter_df

def update_age(i, fighter_df):
    # Get Age
    birth_year = fighter_df.iloc[i]['born_year']
    date = fighter_df.iloc[i]['Date']
    # Extract the year from the Timestamp
    year = date.year if not pd.isna(date) else None

    if birth_year != '--':
        fighter_df.loc[fighter_df.index[i], 'Age'] = int(year) - int(birth_year)
    else:
        fighter_df.loc[fighter_df.index[i], 'Age'] = 0

    return fighter_df

def update_streak(i, fighter_df, running_stats):
    # Calculate win streak
    if i != 0:
        if fighter_df.iloc[i-1]['W/L 1'] == 'win':
            if running_stats['streak'] < 0:
                running_stats['streak'] = 1
            else:
                running_stats['streak'] += 1
        else:
            if running_stats['streak'] > 0:
                running_stats['streak'] = -1
            else:
                running_stats['streak'] -= 1

        # Update the streak column
        fighter_df.loc[fighter_df.index[i], 'Streak'] = running_stats['streak']
    
    return fighter_df, running_stats

def update_career_win_loss(i, fighter_df, running_stats):
    # Iteratively go back and solve total career stats
    if fighter_df.iloc[len(fighter_df) - 1 - i]['W/L 1']=='win':
        running_stats['career_w'] -= 1
    else:
        running_stats['career_l'] -= 1

    fighter_df.loc[fighter_df.index[len(fighter_df) - 1 - i], 'Career W'] = running_stats['career_w']
    fighter_df.loc[fighter_df.index[len(fighter_df) - 1 - i], 'Career L'] = running_stats['career_l']
    fighter_df.loc[fighter_df.index[len(fighter_df) - 1 - i], 'Career Fights'] = running_stats['career_l'] + running_stats['career_w']
    if running_stats['career_l'] + running_stats['career_w'] != 0:
        fighter_df.loc[fighter_df.index[len(fighter_df) - 1 - i], 'Career W Perc'] = running_stats['career_w'] / (running_stats['career_l'] + running_stats['career_w']) * 100
    else:
        fighter_df.loc[fighter_df.index[len(fighter_df) - 1 - i], 'Career W Perc'] = 0

    return fighter_df, running_stats

def update_win_loss(i, fighter_df, running_stats):
    ### Update for W, L, Num Fights, and W Perc
    result = fighter_df.iloc[i]['W/L 1']  # Assuming 'W/L 1' is the column that has 'W', 'L', or 'NC'
    if result == 'win':
        running_stats['wins'] += 1
    else:
        running_stats['losses'] += 1
    running_stats['fights'] += 1
    running_stats['w_perc'] = running_stats['wins'] / running_stats['fights']
    fighter_df.loc[fighter_df.index[i + 1], 'W'] = running_stats['wins']
    fighter_df.loc[fighter_df.index[i + 1], 'L'] = running_stats['losses']
    fighter_df.loc[fighter_df.index[i + 1], 'Num Fights'] = running_stats['fights']
    fighter_df.loc[fighter_df.index[i + 1], 'W Perc'] = running_stats['w_perc'] * 100.0

    return fighter_df, running_stats

def update_control_time(i, fighter_df, running_stats):
    # Update control time and total time based on existing columns
    for string in ['Ctrl 1', 'Ctrl 2', 'Time']:
        time_str = fighter_df.iloc[i][string]
        if ':' in time_str:
            minutes, seconds = map(int, time_str.split(':'))
            total_seconds = minutes * 60 + seconds
            if string == 'Time':
                total_seconds += (fighter_df.iloc[i]['Round'] - 1) * 300
            fighter_df.loc[fighter_df.index[i], string] = total_seconds
        else:
            fighter_df.loc[fighter_df.index[i], string] = 0

    # Update running totals for 'CTRL Total' and 'Time Total'
    if 'Ctrl 1' in fighter_df.columns:
        running_stats['running_ctrl_1'] += fighter_df.iloc[i]['Ctrl 1']
    if 'Ctrl 2' in fighter_df.columns:
        running_stats['running_ctrl_2'] += fighter_df.iloc[i]['Ctrl 2']
    if 'Time' in fighter_df.columns:
        running_stats['running_time'] += fighter_df.iloc[i]['Time']

    # Assign running totals to new columns
    fighter_df.loc[fighter_df.index[i+1], 'CTRL Avg'] = 0.0 if running_stats['fights']==0.0 else running_stats['running_ctrl_1']/running_stats['fights']
    fighter_df.loc[fighter_df.index[i+1], 'CTRL Opp Avg'] = 0.0 if running_stats['fights']==0.0 else running_stats['running_ctrl_2']/running_stats['fights']
    fighter_df.loc[fighter_df.index[i+1], 'Time Avg'] = 0.0 if running_stats['fights']==0.0 else running_stats['running_time']/running_stats['fights']

    return fighter_df, running_stats

def update_sig_strike(i, fighter_df, running_stats):
    ### Update for Sig Str 1 totals and running career sig str %
    for j, sig_str_col in enumerate(['Sig. str. 1', 'Sig. str. 2'], start=1):
        sig_str_value = fighter_df.iloc[i][sig_str_col]
        strikes, attempts = map(int, sig_str_value.split(' of '))

        if j == 1:
            running_stats['running_sig_strikes_1'] += strikes
            running_stats['running_attempted_strikes_1'] += attempts
            if running_stats['running_attempted_strikes_1'] == 0:
                running_stats['sig_str_percentage_1'] = 0
            else:
                running_stats['sig_str_percentage_1'] = (running_stats['running_sig_strikes_1'] / running_stats['running_attempted_strikes_1']) * 100

            fighter_df.loc[fighter_df.index[i + 1], 'Sig Strikes Avg'] = 0.0 if running_stats['fights']==0.0 else running_stats['running_sig_strikes_1']/running_stats['fights']
            fighter_df.loc[fighter_df.index[i + 1], 'Sig Str %'] = running_stats['sig_str_percentage_1']

        elif j == 2:
            running_stats['running_sig_strikes_2'] += strikes
            running_stats['running_attempted_strikes_2'] += attempts
            if running_stats['running_attempted_strikes_2'] == 0:
                running_stats['sig_str_percentage_2'] = 0
            else:
                running_stats['sig_str_percentage_2'] = (running_stats['running_sig_strikes_2'] / running_stats['running_attempted_strikes_2']) * 100

            fighter_df.loc[fighter_df.index[i + 1], 'Sig Strikes Opp Avg'] = 0.0 if running_stats['fights']==0.0 else running_stats['running_sig_strikes_2']/running_stats['fights']
            fighter_df.loc[fighter_df.index[i + 1], 'Sig Str % Opp'] = running_stats['sig_str_percentage_2']
    
    return fighter_df, running_stats

def update_strike(i, fighter_df, running_stats):
    ## Update for Total Str (Strikes 1 and Strikes 2)
    for j, total_str_col in enumerate(['Total str. 1', 'Total str. 2'], start=1):
        total_str_value = fighter_df.iloc[i][total_str_col]
        strikes, attempts = map(int, total_str_value.split(' of '))

        if j == 1:
            running_stats['running_strikes_1'] += strikes
            running_stats['running_attempted_strikes_1'] += attempts
            if running_stats['running_attempted_strikes_1'] == 0:
                running_stats['str_percentage_1'] = 0
            else:
                running_stats['str_percentage_1'] = (running_stats['running_strikes_1'] / running_stats['running_attempted_strikes_1']) * 100

            fighter_df.loc[fighter_df.index[i + 1], 'Strikes Avg'] = 0.0 if running_stats['fights']==0.0 else running_stats['running_strikes_1']/running_stats['fights']
            fighter_df.loc[fighter_df.index[i + 1], 'Str %'] = running_stats['str_percentage_1']

        elif j == 2:
            running_stats['running_strikes_2'] += strikes
            running_stats['running_attempted_strikes_2'] += attempts
            if running_stats['running_attempted_strikes_2'] == 0:
                running_stats['str_percentage_2'] = 0
            else:
                running_stats['str_percentage_2'] = (running_stats['running_strikes_2'] / running_stats['running_attempted_strikes_2']) * 100

            fighter_df.loc[fighter_df.index[i + 1], 'Strikes Avg'] = 0.0 if running_stats['fights']==0.0 else running_stats['running_strikes_2']/running_stats['fights']
            fighter_df.loc[fighter_df.index[i + 1], 'Str %'] = running_stats['str_percentage_2']
    
    return fighter_df, running_stats

def update_td(i, fighter_df, running_stats):
    for j, td_str_col in enumerate(['Td 1', 'Td 2'], start=1):
        td_str = fighter_df.iloc[i][td_str_col]
        td, attempts = map(int, td_str.split(' of '))

        if j == 1:
            running_stats['running_td_1'] += td
            running_stats['running_attempted_td_1'] += attempts
            if running_stats['running_attempted_td_1'] == 0:
                running_stats['td_percentage_1'] = 0
            else:
                running_stats['td_percentage_1'] = (running_stats['running_td_1'] / running_stats['running_attempted_td_1']) * 100

            fighter_df.loc[fighter_df.index[i + 1], 'TD Avg'] = 0.0 if running_stats['fights']==0.0 else running_stats['running_td_1']/running_stats['fights']
            fighter_df.loc[fighter_df.index[i + 1], 'TD %'] = running_stats['td_percentage_1']

        elif j == 2:
            running_stats['running_td_2'] += td
            running_stats['running_attempted_td_2'] += attempts
            if running_stats['running_attempted_td_2'] == 0:
                running_stats['td_percentage_2'] = 0
            else:
                running_stats['td_percentage_2'] = (running_stats['running_td_2'] / running_stats['running_attempted_td_2']) * 100

            fighter_df.loc[fighter_df.index[i + 1], 'TD Avg'] = 0.0 if running_stats['fights']==0.0 else running_stats['running_td_2']/running_stats['fights']
            fighter_df.loc[fighter_df.index[i + 1], 'TD %'] = running_stats['td_percentage_2']
    
    return fighter_df, running_stats

def update_kd(i, fighter_df, running_stats):
    ### Update for KD Total 1 and KD Total 2
    for j, kd_col in enumerate(['KD 1', 'KD 2'], start=1):
        kd = fighter_df.iloc[i][kd_col]

        if j == 1:
            running_stats['running_kd_1'] += kd
            fighter_df.loc[fighter_df.index[i + 1], 'KD Avg'] = 0.0 if running_stats['fights']==0.0 else running_stats['running_kd_1']/running_stats['fights']

        elif j == 2:
            running_stats['running_kd_2'] += kd
            fighter_df.loc[fighter_df.index[i + 1], 'KD Opp Avg'] = 0.0 if running_stats['fights']==0.0 else running_stats['running_kd_2']/running_stats['fights']

    return fighter_df, running_stats

def update_method(i, fighter_df, running_stats):
    ### Update for Method DEC and KO
    method = fighter_df.iloc[i]['Method']
    if 'DEC' in method:
        if fighter_df.iloc[i]['W/L 1'] == 'win':
            running_stats['running_dec'] += 1
        else:
            running_stats['running_dec_opp'] += 1
    if 'KO' in method:
        if fighter_df.iloc[i]['W/L 1'] == 'win':
            running_stats['running_ko'] += 1
        else:
            running_stats['running_ko_opp'] += 1
    if 'SUB' in method:
        if fighter_df.iloc[i]['W/L 1'] == 'win':
            running_stats['running_sub'] += 1
        else:
            running_stats['running_sub_opp'] += 1

    # Update dataframe
    fighter_df.loc[fighter_df.index[i + 1], 'DEC Avg'] = 0.0 if running_stats['fights']==0.0 else running_stats['running_dec']/running_stats['fights']
    fighter_df.loc[fighter_df.index[i + 1], 'DEC Opp Avg'] = 0.0 if running_stats['fights']==0.0 else running_stats['running_dec_opp']/running_stats['fights']
    fighter_df.loc[fighter_df.index[i + 1], 'KO Avg'] = 0.0 if running_stats['fights']==0.0 else running_stats['running_ko']/running_stats['fights']
    fighter_df.loc[fighter_df.index[i + 1], 'KO Opp Avg'] = 0.0 if running_stats['fights']==0.0 else running_stats['running_ko_opp']/running_stats['fights']
    fighter_df.loc[fighter_df.index[i + 1], 'SUB Avg'] = 0.0 if running_stats['fights']==0.0 else running_stats['running_sub']/running_stats['fights']
    fighter_df.loc[fighter_df.index[i + 1], 'SUB Opp Avg'] = 0.0 if running_stats['fights']==0.0 else running_stats['running_sub_opp']/running_stats['fights']

    return fighter_df, running_stats

def create_new_stats(fighter_dfs, more_fighter_stats):
    for fighter in fighter_dfs:
        fighter_df = fighter_dfs[fighter]

        # Initialize all the new values with zeros
        fighter_df = initialize_columns(fighter_df)

        running_stats = initialize_running_stats(fighter_df)

        for i in range(len(fighter_df)):
            ### Update Stats
            fighter_df = update_physical(i, fighter_df, more_fighter_stats, fighter)
            fighter_df = update_age(i, fighter_df)
            fighter_df, running_stats = update_streak(i, fighter_df, running_stats)
            fighter_df, running_stats = update_career_win_loss(i, fighter_df, running_stats)

            # Skip last loop for other stats
            if i == (len(fighter_df) - 1):
                continue
            
            # Update next fight stats
            fighter_df, running_stats = update_win_loss(i, fighter_df, running_stats)
            fighter_df, running_stats = update_control_time(i, fighter_df, running_stats)
            fighter_df, running_stats = update_sig_strike(i, fighter_df, running_stats)
            fighter_df, running_stats = update_strike(i, fighter_df, running_stats)
            fighter_df, running_stats = update_td(i, fighter_df, running_stats)
            fighter_df, running_stats = update_kd(i, fighter_df, running_stats)
            fighter_df, running_stats = update_method(i, fighter_df, running_stats)

        # Remove the specified columns
        fighter_df = clean_columns(fighter_df)

        # Add df to dictionary
        fighter_dfs[fighter] = fighter_df

        if fighter == 'Gilbert Burns':
            print(fighter_df)
            fighter_df.to_csv("Data/burns_new_stats.csv")
    
    return fighter_dfs

def combine_all_fights(fighter_dfs):
    # Initialize an empty list to collect all the DataFrames
    all_fights_list = []
    # Iterate over the dictionary of fighter DataFrames
    for fighter, fighter_df in fighter_dfs.items():
        # Append the fighter's DataFrame to the list
        all_fights_list.append(fighter_df)
    # Concatenate all the DataFrames in the list into one large DataFrame
    all_fights_combined = pd.concat(all_fights_list, ignore_index=True)
    # Optionally, drop any duplicate rows to ensure each fight is only listed once
    return all_fights_combined.drop_duplicates()

def combine_fighter_stats(all_fights_combined):
    # List of unique stats
    unique_stats = ['W', 'L', 'Num Fights', 'W Perc', 'Sig Strikes Avg', 'Sig Str %',
                    'Sig Strikes Opp Avg', 'Sig Str % Opp', 'Strikes Avg', 'Str %',
                    'Strikes Opp Avg', 'Str % Opp', 'TD Avg', 'TD %', 'TD Opp Avg',
                    'TD % Opp', 'KD Avg', 'KD Opp Avg', 'DEC Avg', 'KO Avg',
                    'SUB Avg', 'DEC Opp Avg', 'KO Opp Avg', 'SUB Opp Avg', 'CTRL Avg',
                    'CTRL Opp Avg', 'Time Avg', 'Streak', 'Career W', 'Career L', 'Career Fights', 'Career W Perc',
                    'Ht Diff', 'Reach Diff', 'Age', 'Stance','born_year']

    # Create a new DataFrame to store combined rows
    combined_df = pd.DataFrame()

    # Track processed indices to avoid re-processing
    processed_indices = set()

    # Iterate over each row to find and combine matching rows
    for idx, row1 in all_fights_combined.iterrows():
        # Skip if this row has already been processed
        if idx in processed_indices:
            continue

        # Find the matching row where Fighter 1 and Fighter 2 are swapped
        matching_row = all_fights_combined[
            (all_fights_combined['Event'] == row1['Event']) &
            (all_fights_combined['Fighter 1'] == row1['Fighter 2']) &
            (all_fights_combined['Fighter 2'] == row1['Fighter 1'])
        ]

        if not matching_row.empty:
            row2 = matching_row.iloc[0]
            row2_idx = matching_row.index[0]

            # Combine rows, giving '1' suffix to row1 stats and '2' suffix to row2 stats
            combined_row = row1.copy()

            # Update stats with '2' suffix from the second row
            for stat in unique_stats:
                combined_row[stat + ' 2'] = row2[stat]

            # Rename original stats to '1' suffix for consistency
            for stat in unique_stats:
                combined_row.rename({stat: stat + ' 1'}, inplace=True)

            # Add combined row to the new DataFrame
            combined_df = pd.concat([combined_df, combined_row.to_frame().T], ignore_index=True)

            # Add the indices of processed rows to avoid duplication
            processed_indices.add(idx)
            processed_indices.add(row2_idx)

    # Reset index for the final combined DataFrame
    combined_df.reset_index(drop=True, inplace=True)

    # Replace the NaN with None
    combined_df['Stance 1'] = combined_df['Stance 1'].replace({np.nan: None})
    combined_df['Stance 2'] = combined_df['Stance 2'].replace({np.nan: None})   

    return combined_df

def ensure_winner_is_fighter_1(df):
    # Define columns that need to be swapped, excluding 'Fighter 1' and 'Fighter 2'
    stat_columns = [col for col in df.columns if (col.endswith('1') or col.endswith('2')) and 'Fighter' not in col]

    for index, row in df.iterrows():
        # If Fighter 2 is the winner, swap Fighter 1 and Fighter 2 and their corresponding stats
        if row['Winner'] == 0:
            # Temporarily store Fighter 1 and Fighter 2
            fighter_1 = row['Fighter 1']
            fighter_2 = row['Fighter 2']
            
            # Swap fighters
            df.at[index, 'Fighter 1'] = fighter_2
            df.at[index, 'Fighter 2'] = fighter_1

            # Set Winner to 1 since Fighter 1 is now the winner
            df.at[index, 'Winner'] = 1

            # Swap all corresponding stats
            for stat in stat_columns:
                if stat.endswith('1'):
                    # Swap stats using temporary variables
                    stat_1 = stat
                    stat_2 = stat[:-1] + '2'
                    temp_stat_1 = row[stat_1]
                    temp_stat_2 = row[stat_2]
                    df.at[index, stat_1] = temp_stat_2
                    df.at[index, stat_2] = temp_stat_1

    return df

def randomize_winner(df):
    # Define columns that need to be swapped, excluding 'Fighter 1' and 'Fighter 2'
    stat_columns = [col for col in df.columns if (col.endswith('1') or col.endswith('2')) and 'Fighter' not in col]

    for index, row in df.iterrows():
        # Randomly decide whether to flip (50% chance)
        flip_winner = np.random.choice([0, 1])  # 0 or 1 with equal probability

        if flip_winner == 1:
            # Temporarily store Fighter 1 and Fighter 2
            fighter_1 = row['Fighter 1']
            fighter_2 = row['Fighter 2']

            # Swap fighters
            df.at[index, 'Fighter 1'] = fighter_2
            df.at[index, 'Fighter 2'] = fighter_1

            # Swap winner
            df.at[index, 'Winner'] = 1 if row['Winner'] == 0 else 0

            # Swap all corresponding stats using temporary variables
            for stat in stat_columns:
                if stat.endswith('1'):
                    stat_1 = stat
                    stat_2 = stat[:-1] + '2'
                    temp_stat_1 = row[stat_1]
                    temp_stat_2 = row[stat_2]
                    df.at[index, stat_1] = temp_stat_2
                    df.at[index, stat_2] = temp_stat_1

    return df

def clean_bouts_data(bouts):
    print("bouts columns ", bouts.columns)
    bouts['W/L 1'] = bouts['W/L 1'].fillna('loss')
    bouts = bouts.loc[bouts["W/L 1"].isin(["win", "loss"])].copy()
    # bouts.drop(["Unnamed: 0"], inplace=True, axis=1)
    bouts['Winner'] = bouts.apply(lambda row: row['Fighter 1'] if row['W/L 1'] == 'win' else row['Fighter 2'], axis=1)
    columns = ['Winner'] + [col for col in bouts.columns if col != 'Winner']
    bouts = bouts[columns]  
    bouts.drop(["W/L 1"], inplace=True, axis=1)
    bouts.drop("born_year 1", axis=1, inplace=True)
    bouts.drop("born_year 2", axis=1, inplace=True)
    bouts["Winner"] = bouts["Winner"] == bouts["Fighter 1"]
    bouts["Winner"] = bouts["Winner"].astype(int)
    bouts["Winner"].value_counts()
    return bouts

def prepare_data_for_analysis(combined_df):    
    bouts = combined_df

    # Clean data
    print("columns before clean ", bouts.columns)
    bouts = clean_bouts_data(bouts)
    print("columns after clean ", bouts.columns)

    ## Randomize winner
    bouts = ensure_winner_is_fighter_1(bouts)
    bouts = randomize_winner(bouts)

    bouts.reset_index(inplace=True, drop=True)

    return bouts

def main():
    # Get fighter and bouts data
    fighters = pd.read_csv("Data/fighters15.csv")
    bouts_clean = pd.read_csv("Data/bouts_913.csv")

    # Edit columns
    fighters = drop_edit_col_names(fighters)

    # print(fighters[fighters.duplicated(subset="Name", keep=False)])
    fighters, bouts_clean = fix_duplicates(fighters, bouts_clean)

    # Invidual fighter stats
    more_fighter_stats = clean_fighter_stats(fighters)

    # Find all unique fighter names
    unique_fighters = pd.unique(bouts_clean[['Fighter 1', 'Fighter 2']].values.ravel('K'))

    # Create dictionary of each fighters bout history
    fighter_dfs = create_fighter_dictionaries(bouts_clean, unique_fighters, more_fighter_stats)
    
    # Create new stats for data analysis
    fighter_dfs = create_new_stats(fighter_dfs, more_fighter_stats)

    # Now combine both fighters stats
    all_fights_combined = combine_all_fights(fighter_dfs)
    combined_df = combine_fighter_stats(all_fights_combined)

    combined_df.to_csv("Data/combined_df_927.csv")

    # Back to processing before analysis
    bouts = prepare_data_for_analysis(combined_df)
    bouts.to_csv("Data/ufc_combined_0927_2.csv")

if __name__ == "__main__":
    main()