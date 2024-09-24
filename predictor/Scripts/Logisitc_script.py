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

def find_probabilities(logreg, X_train_scaled, X_train, y_train, X_predict):
    # Fit the model (assuming X_train and y_train are already defined)
    logreg.fit(X_train_scaled, y_train)

    # Get the coefficients
    coefficients = logreg.coef_[0]
    feature_names = X_train.columns

    # Create a DataFrame to view the coefficients with feature names
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
    coef_df['AbsCoefficient'] = coef_df['Coefficient'].abs()  # Absolute value for comparison
    coef_df = coef_df.sort_values(by='AbsCoefficient', ascending=False)

    intercept = logreg.intercept_

    # coef_df.to_csv("coefficients_922_new.csv")

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

    return X_train_scaled, X_train, y_train, logreg, y_pred, X_test, y_test

def find_bets(odds, fights, probabilities):
    total_money = 100
    winnings = 0

    print("starting money ", total_money)
    bets = []
    for i in range(len(odds)):
        # Get probabilities and odds
        prob = probabilities[i]
        odd1 = odds.iloc[i]['Fighter 1 Odds']
        odd2 = odds.iloc[i]['Fighter 2 Odds']

        # Find decimal odds for who to bet on
        if (prob >= 0.5).astype(int):
            b = 1+ (odd1/100)
        else:
            b = 1 + (odd2/100)
        
        # Calculate percentage bet
        f = (b*prob - 1 + prob)/b
        print("Percent to bet ", f)
        bet = f*total_money/100
        bets.append(bet)
        print("Bet ", bet)

        if (prob >= 0.5).astype(int) ==  fights['Winner'].to_list()[i]:
            print("money won ", bet)
            winnings += bet
        else:
            print("money lost ", bet)
            winnings -= bet
        
    print("winnings ", winnings)
    return bets

def calc_stat_importance(odds):
    # Assuming you have a DataFrame called df with your features and target variable
    # Let's say your target variable is 'Winner'
    stats = ['Ht Diff', 'Reach Diff', 'Career W Diff', 'Career L Diff',
        'W Diff', 'L Diff', 'Num Fights Diff', 'W Perc Diff',
        'Sig Strikes Avg Diff', 'Sig Str % Diff', 'Sig Strikes Opp Avg Diff',
        'Sig Str % Opp Diff', 'Strikes Avg Diff', 'Str % Diff',
        'Strikes Opp Avg Diff', 'Str % Opp Diff', 'TD Avg Diff', 'TD % Diff',
        'TD Opp Avg Diff', 'TD % Opp Diff', 'KD Avg Diff', 'KD Opp Avg Diff',
        'DEC Avg Diff', 'KO Avg Diff', 'SUB Avg Diff', 'DEC Opp Avg Diff',
        'KO Opp Avg Diff', 'SUB Opp Avg Diff', 'CTRL Avg Diff',
        'CTRL Opp Avg Diff', 'Time Avg Diff', 'Streak Diff', 'Age Diff',
        'Career Fights Diff', 'Career W Perc Diff', 'Stance Diff']

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
        'Career W', 'Career L', 'W', 'L', 'Num Fights',
        'W Perc', 'Sig Strikes Avg', 'Sig Str %', 'Sig Strikes Opp Avg',
        'Sig Str % Opp', 'Strikes Avg', 'Str %', 'Strikes Opp Avg',
        'Str % Opp', 'TD Avg', 'TD %', 'TD Opp Avg', 'TD % Opp',
        'KD Avg', 'KD Opp Avg', 'DEC Avg', 'KO Avg', 'SUB Avg',
        'DEC Opp Avg', 'KO Opp Avg', 'SUB Opp Avg', 'CTRL Avg',
        'CTRL Opp Avg', 'Time Avg', 'Streak', 'Age',
        'Career Fights', 'Career W Perc', 'Stance'
    ]

    # Create difference columns and drop the original columns
    for stat in stats:
        diff_col_name = f"{stat} Diff"
        fights[diff_col_name] = fights[f"{stat} 1"] - fights[f"{stat} 2"]

    # Now drop the original columns
    cols_to_drop = [f"{stat} 1" for stat in stats] + [f"{stat} 2" for stat in stats]
    fights.drop(columns=cols_to_drop, inplace=True)

    fights.drop('Date', axis=1, inplace=True)
    fights.drop('Fighter 1 Odds', axis=1, inplace=True)
    fights.drop('Fighter 2 Odds', axis=1, inplace=True)
    fights.drop('Location', axis=1, inplace=True)
    fights.drop('Fighter 1', axis=1, inplace=True)
    fights.drop('Fighter 2', axis=1, inplace=True)
    fights.drop('Event', axis=1, inplace=True)
    fights.drop('Weight class', axis=1, inplace=True)
    fights.drop('Ht Diff 2', axis=1, inplace=True)
    fights.drop('Reach Diff 2', axis=1, inplace=True)
    fights.rename(columns={'Ht Diff 1': 'Ht Diff'}, inplace=True)
    fights.rename(columns={'Reach Diff 1': 'Reach Diff'}, inplace=True)

# This is the main function
def main(event):
    # Get odds csv and drop useless columns
    odds = pd.read_csv("ufc_combined_money_921_date.csv", index_col=0)
    odds.drop('born_year 1', axis=1, inplace=True)
    odds.drop('born_year 2', axis=1, inplace=True)

    # Get odds and winners from Event to predict
    fights_predict = odds[odds['Event'].str.contains(event, case=False, na=False)]
    odds_predict = fights_predict[['Fighter 1 Odds', 'Fighter 2 Odds']]
    edit_data(fights_predict)
    X = fights_predict.drop(columns=['Winner'])  # Features (all columns except 'Winner')
    scaler = StandardScaler()
    X_predict = scaler.fit_transform(X)

    # Edit data for odds/fights
    edit_data(odds)

    # Predict fights
    X_train_scaled, X_train, y_train, logreg, y_pred, X_test, y_test = predict_fights(odds)

    # Some data analysis of regression
    plot_cnf(y_test, y_pred)
    get_classification(y_test, y_pred)
    plot_auc(X_test, y_test,logreg)

    # Assuming y_test are the true labels and y_pred are the predicted labels
    find_accuracy(y_test, y_pred)

    # Find probabilites for each fight
    probabilities = find_probabilities(logreg, X_train_scaled, X_train, y_train, X_predict)

    # Find bets
    bets = find_bets(odds_predict, fights_predict, probabilities)

# Entry point of the script
if __name__ == "__main__":
    # Find bets for a UFC event
    main('UFC 303')
