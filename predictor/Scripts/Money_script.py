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

            # Combine the rows by merging dictionaries
            combined_row = {**row1.to_dict(), **match.to_dict()}

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

def edit_dataframes(fights_df, odds_df):
    fights_df['Date'] = fights_df['Date'].str[:4]
    fights_df['Date'] = fights_df['Date'].astype(int)

    odds_df['Fighter 1'] = odds_df['Fighter']
    odds_df['Fighter 2'] = odds_df['Opponent']
    odds_df['Fighter 1 Odds'] = odds_df['Fighter Odds']
    odds_df['Fighter 2 Odds'] = odds_df['Opponent Odds']
    odds_df.drop('Fighter', axis=1, inplace=True)
    odds_df.drop('Fighter Odds', axis=1, inplace=True)
    odds_df.drop('Opponent', axis=1, inplace=True)
    odds_df.drop('Opponent Odds', axis=1, inplace=True)
    odds_df['Date'] = odds_df['Date'].str[-4:]
    odds_df['Date'] = odds_df['Date'].astype(int)

def main():
    odds_df = pd.read_csv('Data/combined_fight_odds_916.csv', index_col=0)
    fights_df = pd.read_csv('Data/ufc_combined_0923_2.csv', index_col=0)   
    
    edit_dataframes(fights_df, odds_df)

    combined_df, other_df = find_fights_with_odds(fights_df, odds_df)

    combined_df.to_csv("ufc_combined_money_923_date.csv")

if __name__ == "__main__":
    main()
