import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score

def find_accuracy(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

def find_coefficients(logreg, X_train_scaled, X_train, y_train): #, X_predict):
    # Fit the model (assuming X_train and y_train are already defined)
    logreg.fit(X_train_scaled, y_train)

    # Get the coefficients
    coefficients = logreg.coef_[0]
    feature_names = X_train.columns

    # Create a DataFrame to view the coefficients with feature names
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
    coef_df['AbsCoefficient'] = coef_df['Coefficient'].abs()  # Absolute value for comparison
    coef_df = coef_df.sort_values(by='AbsCoefficient', ascending=False)
    print("coef_df ", coef_df)

    return coefficients

def find_probabilities(logreg, X_predict, coefficients):
    # Step 1: Get intercept
    intercept = logreg.intercept_
    # Step 2: Calculate logits
    logits = intercept + np.dot(X_predict, coefficients)
    # Step 3: Convert logits to probabilities using the logistic function
    return 1 / (1 + np.exp(-logits))

def plot_auc(X_test, y_test, logreg):
    y_pred_proba = logreg.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.legend(loc=4)
    plt.show()

def get_classification(y_test, y_pred):
    target_names = ['loser', 'winner']
    print("classification_report")
    print(classification_report(y_test, y_pred, target_names=target_names))

def plot_cnf(y_test, y_pred):
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    class_names=[0,1] # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

def predict_fights(odds):
    print("len(odds) ",len(odds))
    # Split the data into features (X) and target (y)
    X = odds.drop(columns=['Winner'])  # Features (all columns except 'Winner')
    y = odds['Winner']                 # Target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

    # Scale the data after splitting
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Fit the model with the scaled training data and corresponding target
    logreg = LogisticRegression(random_state=16)
    logreg.fit(X_train_scaled, y_train)  # Ensure you're using y_train here

    # Predict with the test data
    y_pred = logreg.predict(X_test_scaled)

    print("len(y_pred) ", len(y_pred))
    print("y_pred ", y_pred)

    return X_train_scaled, X_train, y_train, logreg, y_pred, X_test, y_test

def find_bets(odds, fights, probabilities):
    total_money = 10
    bets = []
    profit = 0 #For fights that have happened
    winnings = [] #For fights that have happened
    print("all odds ", odds)
    print("actual winners ",fights['Winner'].to_list())
    print("probabilities ", probabilities)
    print("starting money ", total_money)
    for i in range(len(odds)):
        print("")
        # Get probabilities and odds
        prob = probabilities[i]
        odd1 = odds.iloc[i]['Fighter 1 Odds']
        odd2 = odds.iloc[i]['Fighter 2 Odds']
        print("odd1 ", odd1)
        print("odd2 ", odd2)
        print("fighter 1 ", odds.iloc[i]['Fighter 1'])
        print("fighter 2 ", odds.iloc[i]['Fighter 2'])
        print("winner ", fights['Winner'].to_list()[i])

        # Find decimal odds for who to bet on
        if (prob >= 0.5).astype(int):
            print("predction: winner=1")
            dec_odds = -100.0/odd1 + 1.0 if odd1 < 0.0 else odd1/100.0 + 1.0
            my_winner = 1
        else:
            print("predction: winner=0")
            dec_odds = -100.0/odd2 + 1.0 if odd2 < 0 else odd2/100.0 + 1.0
            my_winner = 0
            prob = 1 - prob
        
        print("my winner prob ", prob)
        print("decimal odds ", dec_odds)
        
        # Calculate percentage bet
        fraction = prob - (1-prob)/(dec_odds-1)
        print("Percent to bet ", fraction)

        bet = fraction*total_money
        if bet < 0:
            print("bet is negative so don't bet, too risky ", bet)
            bet = 0
        bets.append(bet)
        print("Bet ", bet)

        print("if i win, here is money back ", bet*dec_odds - bet)


        ### Only use this for fights that have HAPPENED
        if my_winner == fights['Winner'].to_list()[i]:
            money_won = bet*dec_odds - bet
            print("money won ", money_won)
            profit += money_won
        else:
            money_won = -bet
            print("money lost ", money_won)
            profit += money_won

        winnings.append(money_won)
        
    print("profit ", profit)
    return bets, profit

def calc_stat_importance(odds):
    # Assuming you have a DataFrame called df with your features and target variable
    # Let's say your target variable is 'Winner'
    stats = ['Ht Diff', 'Reach Diff', 'Career W Diff', 'Career L Diff',
        'W Diff', 'L Diff', 'Num Fights Diff', 'W Perc Diff',
        'Sig Strikes Avg Diff', 'Sig Str % Diff', 'Sig Strikes Opp Avg Diff',
        'Sig Str % Opp Diff', 'Strikes Avg Diff', 'Str % Diff',
        'Strikes Opp Avg Diff', 'Str % Opp Diff', 'TD Avg Diff', 'TD % Diff',
        'TD Opp Avg Diff', 'TD % Opp Diff', 'KD Avg Diff', 'KD Opp Avg Diff',
        'KO Avg Diff', 'SUB Avg Diff',
        'KO Opp Avg Diff', 'SUB Opp Avg Diff', 'CTRL Avg Diff',
        'CTRL Opp Avg Diff', 'Time Avg Diff', 'Streak Diff', 'Age Diff',
         'Stance Diff', 'Career Fights Diff', 'Career W Perc Diff']

    # Dictionary to hold accuracies
    accuracy_results = {}
    for stat in stats:
        # Create a DataFrame with just the stat and the target variable
        X = odds[[stat]]  # Features
        y = odds['Winner']  # Target variable
        
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and fit the logistic regression model
        logreg = LogisticRegression(random_state=16)
        logreg.fit(X_train, y_train)

        # Predict on the test set
        y_pred = logreg.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_results[stat] = accuracy

    # Convert results to a DataFrame for better visualization
    accuracy_df = pd.DataFrame(list(accuracy_results.items()), columns=['Stat', 'Accuracy'])
    accuracy_df.sort_values(by='Accuracy', ascending=False, inplace=True)
    print(accuracy_df)

def edit_data(fights):
    cats = ["Fighter 1", "Fighter 2", "Weight class", "Stance 1", "Stance 2", "Location", "Event"]

    for cat in cats:
        fights[cat] = fights[cat].fillna("unk").astype("category")

    fights[cats] = fights[cats].apply(lambda col: col.cat.codes)

    stats = [
        'W', 'L', 'Num Fights', 'Career W', 'Career L', 
        'W Perc', 'Sig Strikes Avg', 'Sig Str %', 'Sig Strikes Opp Avg',
        'Sig Str % Opp', 'Strikes Avg', 'Str %', 'Strikes Opp Avg',
        'Str % Opp', 'TD Avg', 'TD %', 'TD Opp Avg', 'TD % Opp',
        'KD Avg', 'KD Opp Avg', 'KO Avg', 'SUB Avg',
        'KO Opp Avg', 'SUB Opp Avg', 'CTRL Avg',
        'CTRL Opp Avg', 'Time Avg', 'Streak', 'Age',
         'Stance' ,'Career Fights', 'Career W Perc'
    ]

    # Create difference columns and drop the original columns
    for stat in stats:
        diff_col_name = f"{stat} Diff"
        fights[diff_col_name] = fights[f"{stat} 1"] - fights[f"{stat} 2"]

    # Now drop the original columns
    cols_to_drop = [f"{stat} 1" for stat in stats] + [f"{stat} 2" for stat in stats]
    fights.drop(columns=cols_to_drop, inplace=True)

    fights.drop('DEC Avg 1', axis=1, inplace=True)
    fights.drop('DEC Opp Avg 1', axis=1, inplace=True)
    fights.drop('DEC Avg 2', axis=1, inplace=True)
    fights.drop('DEC Opp Avg 2', axis=1, inplace=True)

    fights.drop('Date', axis=1, inplace=True)
    # fights.drop('Fighter 1 Odds', axis=1, inplace=True)
    # fights.drop('Fighter 2 Odds', axis=1, inplace=True)
    fights.drop('Location', axis=1, inplace=True)
    fights.drop('Fighter 1', axis=1, inplace=True)
    fights.drop('Fighter 2', axis=1, inplace=True)
    fights.drop('Event', axis=1, inplace=True)
    fights.drop('Weight class', axis=1, inplace=True)
    fights.drop('Ht Diff 2', axis=1, inplace=True)
    fights.drop('Reach Diff 2', axis=1, inplace=True)
    fights.rename(columns={'Ht Diff 1': 'Ht Diff'}, inplace=True)
    fights.rename(columns={'Reach Diff 1': 'Reach Diff'}, inplace=True)
    return fights

def align_odds(odds_event, fights_event):
    # Create a mask to keep track of which fights to keep in fights_event
    fights_to_keep = []
    odds_to_keep = []

    for i, row in fights_event.iterrows():
        fighter1 = row['Fighter 1'].lower()  # Convert to lower case
        fighter2 = row['Fighter 2'].lower()  # Convert to lower case

        # Find the corresponding row in odds_event with case-insensitive comparison
        mask = (odds_event['Fighter 1'].str.lower().isin([fighter1, fighter2])) & \
               (odds_event['Fighter 2'].str.lower().isin([fighter1, fighter2]))

        if mask.any():  # Check if there is a matching row
            odds_row = odds_event[mask].copy()

            # Reorder Fighter 1 and Fighter 2 in odds_event
            if odds_row['Fighter 1'].values[0].lower() != fighter1:
                odds_event.loc[mask, ['Fighter 1', 'Fighter 2']] = odds_row[['Fighter 2', 'Fighter 1']].values
                odds_event.loc[mask, ['Fighter 1 Odds', 'Fighter 2 Odds']] = odds_row[['Fighter 2 Odds', 'Fighter 1 Odds']].values
            
            # Keep this fight in fights_to_keep
            fights_to_keep.append(i)

            odds_to_keep.extend(odds_event.index[mask].tolist())
        else:
            print("poop ", fighter1)
            print("poop ", fighter2)

    # Drop fights from fights_event that do not have corresponding odds
    fights_event = fights_event.loc[fights_to_keep].reset_index(drop=True)

    # Drop fights from odds_event that do not exist in fights_event
    odds_event = odds_event.loc[odds_to_keep].reset_index(drop=True)

    print("Updated odds_event:\n", odds_event)
    print("Updated fights_event:\n", fights_event)
    
    return odds_event, fights_event


# This is the main function
def main(event):
    # # Get odds csv and drop useless columns
    # odds = pd.read_csv("Data/ufc_combined_money_921_date.csv", index_col=0)
    # fights = pd.read_csv("Data/ufc_combined_money_1004.csv", index_col=0)

    # # Create a new DataFrame for rows where 'Event' is 'UFC 307'
    # fights_307 = fights[fights['Event'] == 'UFC 307'].copy()
    # odds_307 = fights_307[['Fighter 1 Odds', 'Fighter 2 Odds']].copy()
    # fights_307 = edit_data(fights_307)
    # print(fights_307.iloc[0])
    # X = fights_307.drop(columns=['Winner'])  # Features (all columns except 'Winner')
    # scaler = StandardScaler()
    # X_predict = scaler.fit_transform(X)

    # # Drop those rows from the original DataFrame 'bouts'
    # fights = fights[fights['Event'] != 'UFC 307']
    # fights = edit_data(fights)
    # X_train_scaled, X_train, y_train, logreg, y_pred, X_test, y_test = predict_fights(fights)


    #--------
    ### not using Money script
    # fights = pd.read_csv("Data/ufc_combined_0928.csv", index_col=0)
    # fights.drop('Unnamed: 0', axis=1, inplace=True)

    # Separate new fights
    fights = pd.read_csv("Data/ufc_combined_1004.csv", index_col=0)
    fights['Fighter 1'] = fights['Fighter 1'].str.replace('-', ' ', regex=False)
    fights['Fighter 2'] = fights['Fighter 2'].str.replace('-', ' ', regex=False)
    fights['Fighter 1'] = fights['Fighter 1'].str.replace('.', '', regex=False)
    fights['Fighter 2'] = fights['Fighter 2'].str.replace('.', '', regex=False)
    fights_event = fights[fights['Event'].str.contains(event, na=False)].copy()
    fights = fights[~fights['Event'].str.contains('UFC 307', na=False)]
    # fights = fights[~fights['Event'].str.contains(event, na=False)]
    print("fights_event ", fights_event)

    # Get odds for newest fights
    odds = pd.read_csv("Data/ufc_combined_money_1004.csv", index_col=0)
    odds_event = odds[odds['Event'].str.contains(event, na=False)].copy()
    odds_event = odds_event[['Fighter 1','Fighter 2','Fighter 1 Odds','Fighter 2 Odds']]
    print("odds_event ", odds_event)
    odds_event, fights_event = align_odds(odds_event,fights_event)

    # Prepare fights for predicting history
    fights = edit_data(fights)
    X_train_scaled, X_train, y_train, logreg, y_pred, X_test, y_test = predict_fights(fights)

    if not fights_event.empty:
        # Prepare event stats for prediction
        fights_event = edit_data(fights_event)
        X = fights_event.drop(columns=['Winner'])  # Features (all columns except 'Winner')
        scaler = StandardScaler()
        X_predict = scaler.fit_transform(X)

        ### --------
        ### Analysis
        # Some data analysis of regression
        # plot_cnf(y_test, y_pred)
        # get_classification(y_test, y_pred)
        # plot_auc(X_test, y_test,logreg)
        calc_stat_importance(fights)

        # Assuming y_test are the true labels and y_pred are the predicted labels
        find_accuracy(y_test, y_pred)

        #Find coefficients for each stat
        coefficients = find_coefficients(logreg, X_train_scaled, X_train, y_train) #, X_predict)

        #Find probabilites for specific fights
        probabilities = find_probabilities(logreg, X_predict, coefficients)
        print(probabilities)
        print(odds_event)

        # Find bets
        bets, profit = find_bets(odds_event, fights_event, probabilities)

        return profit, sum(bets)
    else:
        print("this event is fucked ", event)
        return 0, 0

# Entry point of the script
if __name__ == "__main__":
    # events = ['UFC ' + str(i) for i in range(200,270)] # total profit  708.3983954721688
    events = ['UFC ' + str(i) for i in range(307,308)] # total profit  708.3983954721688
    total_profit = 0
    total_bet = 0
    for event in events:
        # Find bets for a UFC event
        profit, bet = main(event)
        total_profit += profit
        total_bet += bet

    print("total_bet ", total_bet)
    print("total profit ", total_profit)
