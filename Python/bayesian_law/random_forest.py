# Pandas is used for data manipulation
import pandas as pd
import numpy as np
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
    # Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
import matplotlib.pyplot as plt

def rf(self):
    features = self.dataframeProcessing
    features.describe()

    # One-hot encode the data using pandas get_dummies
    features = pd.get_dummies(features)
    features.iloc[:,5:].head(5)


    # Labels are the values we want to predict
    labels = np.array(features['new_positive'])
    # Remove the labels from the features
    # axis 1 refers to the columns
    features= features.drop('new_positive', axis = 1)
    # Saving feature names for later use
    feature_list = list(features.columns)
    # Convert to numpy array
    features = np.array(features)
    print(features.shape)
    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)

    # The baseline predictions are the historical averages
    # baseline_preds = test_features[:, feature_list.index('average')]
    # Baseline errors, and display average baseline error
    # baseline_errors = abs(baseline_preds - test_labels)
    # print('Average baseline error: ', round(np.mean(baseline_errors), 2))


    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    # Train the model on training data
    rf.fit(train_features, train_labels);
    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    # Calculate the absolute errors
    errors = abs(predictions - test_labels)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'num positive')

    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / test_labels)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')


    # Pull out one tree from the forest
    tree = rf.estimators_[5]
    # Import tools needed for visualization
    from sklearn.tree import export_graphviz
    import pydot
    # Pull out one tree from the forest
    tree = rf.estimators_[5]
    # Export the image to a dot file
    export_graphviz(tree, out_file = 'Data/tree.dot', feature_names = feature_list, rounded = True, precision = 1)
    # Use dot file to create a graph
    (graph, ) = pydot.graph_from_dot_file('Data/tree.dot')
    # Write graph to a png file
    graph.write_png('Plot/tree.png')

    # Get numerical feature importances
    importances = list(rf.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    # Print out the feature and importances
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
    # New random forest with only the two most important variables
    rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=42)
    # Extract the two most important features
    important_indices = [feature_list.index('confirmed_cases_7_day_ago'), feature_list.index('growth_in_3_day')]
    train_important = train_features[:, important_indices]
    test_important = test_features[:, important_indices]
    # Train the random forest
    rf_most_important.fit(train_important, train_labels)
    # Make predictions and determine the error
    predictions = rf_most_important.predict(test_important)
    errors = abs(predictions - test_labels)
    # Display the performance metrics
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'num positive')
    mape = np.mean(100 * (errors / test_labels))
    accuracy = 100 - mape
    print('Accuracy:', round(accuracy, 2), '%.')

    # Import matplotlib for plotting and use magic command for Jupyter Notebooks


    # Set the style
    plt.style.use('fivethirtyeight')
    # list of x locations for plotting
    x_values = list(range(len(importances)))
    # Make a bar chart
    plt.bar(x_values, importances, orientation = 'vertical')
    # Tick labels for x axis
    plt.xticks(x_values, feature_list, rotation='vertical')
    # Axis labels and title
    plt.ylabel('Importance')
    plt.xlabel('Variable')
    plt.title('Variable Importances');
    plt.savefig('plot/varible_Importances')
    plt.close()


    # Use datetime for creating date objects for plotting
    import datetime
    # Dates of training values

    days = features[:, feature_list.index('new_day')]

    # List and then convert to datetime object
    dates = [str(int(day)) for day in days]

    # dates = [datetime.datetime.strptime(date, '%d') for date in dates]
    # Dataframe with true values and dates
    true_data = pd.DataFrame(data = {'date': dates, 'new_positive': labels})
    # # Dates of predictions

    days = test_features[:, feature_list.index('new_day')]

    # Column of dates
    test_dates = [ str(int(day)) for  day in days]
    # Convert to datetime objects
    # test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]
    # Dataframe with predictions and dates
    predictions_data = pd.DataFrame(data = {'date': test_dates, 'prediction': predictions})
    # Plot the actual values
    plt.plot(true_data['date'], true_data['new_positive'], 'b-', label = 'new_positive')
    # Plot the predicted values
    plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label = 'prediction')
    plt.xticks(rotation = '60');
    plt.legend()
    # Graph labels
    plt.xlabel('Date'); plt.ylabel('Maximum Temperature (F)'); plt.title('Actual and Predicted Values');
    plt.savefig('plot/pred.png')
    plt.close()
