import pandas as pd
import numpy as np

def find_fights_with_odds(fights_df, odds_df):
    # Initialize an empty list to store combined rows
    combined_rows = []
    other_rows = []

    # Iterate over each row in fights_df
    for idx1, row1 in fights_df.iterrows():
        # Find the matching row in odds_df
        match = odds_df[((odds_df['Fighter 1'] == row1['Fighter 1']) & (odds_df['Fighter 2'] == row1['Fighter 2']) & (odds_df['Date'] == row1['Date'])) |
                        ((odds_df['Fighter 1'] == row1['Fighter 2']) & (odds_df['Fighter 2'] == row1['Fighter 1']) & (odds_df['Date'] == row1['Date']))]
    
        if not match.empty:
            # Take the first matching row (assuming there is only one match)
            match = match.iloc[0]

            # Combine the rows without overwriting the Winner column from fights_df
            combined_row = {**row1.to_dict(), **match.to_dict()}

            # If fighters are swapped, ensure Winner column is adjusted accordingly
            if (match['Fighter 1'] == row1['Fighter 2']) and (match['Fighter 2'] == row1['Fighter 1']):
                # If fighters are swapped, we need to swap the 'Winner' value
                combined_row['Winner'] = 1 if row1['Winner'] == 0 else 0

            # Append the combined row to the list
            combined_rows.append(combined_row)
        else:
            # If there is no match, append the row from fights_df as is
            other_rows.append(row1.to_dict())

    # Convert the list of combined rows to a DataFrame
    combined_df = pd.DataFrame(combined_rows)
    other_df = pd.DataFrame(other_rows)

    print("\nCombined DataFrame:")
    print(len(combined_df))

    return combined_df, other_df

def remove_duplicates(df):
    # Create new columns to normalize fighter order
    df['Fighter A'] = df.apply(lambda row: min(row['Fighter 1'], row['Fighter 2']), axis=1)
    df['Fighter B'] = df.apply(lambda row: max(row['Fighter 1'], row['Fighter 2']), axis=1)
    
    # Drop duplicates based on 'Fighter A', 'Fighter B', and 'Date'
    df = df.drop_duplicates(subset=['Fighter A', 'Fighter B', 'Date'], keep='first')
    
    # Drop the temporary columns
    df = df.drop(columns=['Fighter A', 'Fighter B'])
    
    return df


def edit_dataframes(fights_df, odds_df):
    odds_df['Fighter 1'] = odds_df['Fighter']
    odds_df['Fighter 2'] = odds_df['Opponent']
    odds_df['Fighter 1 Odds'] = odds_df['Fighter Odds']
    odds_df['Fighter 2 Odds'] = odds_df['Opponent Odds']
    odds_df.drop('Fighter', axis=1, inplace=True)
    odds_df.drop('Fighter Odds', axis=1, inplace=True)
    odds_df.drop('Opponent', axis=1, inplace=True)
    odds_df.drop('Opponent Odds', axis=1, inplace=True)

    # Remove duplicate rows
    odds_df = remove_duplicates(odds_df)

    return fights_df, odds_df

# Function to format date
def format_date(date_str):
    return pd.to_datetime(date_str).strftime('%b %Y')

# Function to convert a string date with 'nth' format and format as "MMM YYYY"
def format_date_to_month_year(date_str):
    # Try to convert the string to datetime
    date = pd.to_datetime(date_str, errors='coerce')
    
    # If the conversion was successful, format it
    if pd.notna(date):
        return date.strftime('%b %Y')
    else:
        print("fuck ", date)
        return None  # or keep the original value by returning `date_str`

def main():
    odds_df = pd.read_csv('Data/combined_fight_odds_916.csv', index_col=0)
    fights_df = pd.read_csv('Data/ufc_combined_0924_2.csv', index_col=0)  
     
    # Apply to the 'Date' column in fights_df
    fights_df['Date'] = fights_df['Date'].apply(format_date)
    # Apply this to the 'Date' column in your dataframe
    odds_df['Date'] = odds_df['Date'].apply(format_date_to_month_year)
    
    fights_df, odds_df = edit_dataframes(fights_df, odds_df)
    fights_df.to_csv("Data/fights_df_check_924.csv")
    odds_df.to_csv("Data/odds_df_check_924.csv")

    combined_df, other_df = find_fights_with_odds(fights_df, odds_df)
    # combined_df['Date'] = combined_df['Date'].str[-4:]
    # combined_df['Date'] = combined_df['Date'].astype(int)

    combined_df.to_csv("Data/ufc_combined_money_924_date.csv")

if __name__ == "__main__":
    main()
