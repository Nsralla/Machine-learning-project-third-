from my_imports import (np, pd, os, StandardScaler, plt, train_test_split,
                        KNeighborsClassifier, accuracy_score, recall_score,
                        precision_score, KFold, SVC, classification_report,
                        LogisticRegression, RandomForestClassifier, confusion_matrix, sns, f1_score)
from sklearn.model_selection import cross_val_score

scaler = StandardScaler()


def read_data_file():
    file_name = 'cardio_train.csv'
    if os.path.exists(file_name):
        # read data
        data = pd.read_csv(file_name, delimiter=';')
        # check if there is any missing values
        if data.isnull().values.any():
            print('There is empty data')
        else:
            print("___________________________________")
            print('No EMPTY DATA IN THE FILE')
            print("___________________________________")
        return data
    else:
        print("___________________________________")
        print("-----File does not exist-----")
        print("___________________________________")
        return None


def data_visualization(data, feature, desc):
    sns.set_style('whitegrid')
    # Plotting the histogram
    plt.figure(figsize=(10, 5))
    sns.histplot(data[feature], bins=100, kde=True)
    plt.title(f'Histogram of {feature}')
    plt.xlabel(desc)
    plt.ylabel('Frequency')
    plt.show()


def clean_data(file_data):
    # drop the id column
    file_data = file_data.drop('id', axis=1)

    min_ap_hi = 90
    max_ap_hi = 200
    min_ap_lo = 60
    max_ap_lo = 120
    # Remove outliers: Keep rows where blood pressure values are within the plausible range
    data_cleaned = remove_blood_outliers(file_data, min_ap_hi, max_ap_hi, min_ap_lo, max_ap_lo)

    # Define plausible ranges for height and weight
    min_height = 110  # cm
    max_height = 220  # cm
    min_weight = 50  # kg
    max_weight = 180  # kg

    # Remove outliers: Keep rows where height and weight values are within the plausible range
    data_cleaned = remove_height_weight_outlier(data_cleaned, min_height, max_height, min_weight, max_weight)

    # converge age to years
    data_cleaned['age'] = round(data_cleaned['age'] / 365.25)

    # check data balance
    print("___________________________________")
    label_balance = data_cleaned['cardio'].value_counts(normalize=True)
    print(f'DATA BALANCE:{label_balance}')
    print("DATA IS BALANCED")
    print("___________________________________")
    # data is balanced

    # scale the following rows:
    # age, weight, height, ap_hi, ap_lo

    data_cleaned = scaling(data_cleaned)

    # one hot encoding for the gender column
    # previously: 1- women, 2- men
    # after encoding: 0-women, 1-men
    data_cleaned = one_hot_encoding(data_cleaned)

    return data_cleaned


def remove_blood_outliers(data, min1, max1, min2, max2):
    data = data[(data['ap_hi'] >= min1) & (data['ap_hi'] <= max1) &
                (data['ap_lo'] >= min2) & (data['ap_lo'] <= max2)]
    return data


def remove_height_weight_outlier(data, min1, max1, min2, max2):
    data = data[
        (data['height'] >= min1) & (data['height'] <= max1) &
        (data['weight'] >= min2) & (data['weight'] <= max2)]
    return data


def choose_Model_with_best_feature(data):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize variables for tracking best features and model
    best_features = []
    best_recall = 0  # Start with 0 to ensure improvement
    best_model = None
    model = None

    # Iterate through each feature, incrementally adding the best one
    for i in range(X.shape[1]):
        remaining_features = list(set(X.columns) - set(best_features))

        # Find the best feature in this iteration
        best_feature_this_iter = None
        best_recall_this_iter = 0

        for feature in remaining_features:
            selected_features = best_features + [feature]
            model = RandomForestClassifier(criterion="entropy",
                                           n_estimators=200,
                                           max_depth=30,
                                           min_samples_split=10,
                                           random_state=5)
            model.fit(X_train[selected_features], y_train)
            predictions = model.predict(X_test[selected_features])
            recall = recall_score(y_test, predictions)

            if recall > best_recall_this_iter:
                best_feature_this_iter = feature
                best_recall_this_iter = recall

        # Update best features and model if there's improvement
        if best_recall_this_iter > best_recall:
            best_features.append(best_feature_this_iter)
            best_recall = best_recall_this_iter
            best_model = model

    # Print the best features and their recall
    print("Best features:", best_features)
    print("Best Recall:", best_recall)

    # Use the best model for predictions or further analysis


def scaling(data):
    columns_to_scale = ['age', 'weight', 'height', 'ap_hi', 'ap_lo']
    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
    return data


def one_hot_encoding(data):
    data = pd.get_dummies(data, columns=['gender'], prefix='gender', drop_first=True)
    # convert from boolean to integer
    data['gender_2'] = data['gender_2'].astype(int)
    # CHANGE THE INDEX OF THE COLUM 'gender_2'
    # Get the list of all column names
    columns = list(data.columns)
    # Remove 'gender_2' from the list
    columns.remove('gender_2')
    columns.insert(1, 'gender_2')
    # Reindex the DataFrame with the new column order
    data = data.reindex(columns=columns)
    return data


def K_NN(X_train, X_test, Y_train, Y_test, k):
    # Initialize K-NN classifier with 3 neighbors
    knn = KNeighborsClassifier(n_neighbors=k)

    # train the model
    knn.fit(X_train, Y_train)

    # test the model
    Y_pred = knn.predict(X_test)

    # Calculate precision and recall and accuracy
    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1Score = f1_score(Y_test, Y_pred)
    confusion = confusion_matrix(Y_test, Y_pred)
    print(f"Precision when K={k}:", precision)
    print("___________________________________")
    print(f"Recall when K={k}:", recall)
    print("___________________________________")
    print(f"Accuracy when K={k}:", accuracy)
    print("___________________________________")
    print(f"F1 score when K={k}:", f1Score)
    print("___________________________________")
    # Print classification report and other metrics
    print(f"Classification Report for KNN, k={k}:")
    print(classification_report(Y_test, Y_pred))
    print("___________________________________")
    # Create a heatmap of the confusion matrix
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix FOR K = {k}")
    plt.show()


def K_NN_with_cross_validation(data):
    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]
    # Initialize variables to store the best k and maximum recall
    best_k = None
    max_recall = 0

    # Iterate over k values from 1 to 30
    for k in range(1, 30):
        recall_values = []  # List to store recall values for each fold

        # Initialize k-fold cross-validation
        kf = KFold(n_splits=15, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            Y_train, Y_val = Y.iloc[train_idx], Y.iloc[val_idx]

            # Create and train the K-NN model
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, Y_train)

            # Make predictions on the validation set
            Y_pred = knn.predict(X_val)

            # Calculate Recall and append it to the list
            recall = recall_score(Y_val, Y_pred)
            recall_values.append(recall)

        # Calculate the average Recall for the current k
        avg_recall = np.mean(recall_values)

        # Check if the current k has a higher average Recall
        if avg_recall > max_recall:
            max_recall = avg_recall
            best_k = k

        print(f'K={k}, Average Recall: {avg_recall}')

    # Print the best k and maximum average Recall
    print(f'Best K: {best_k}, Maximum Average Recall: {max_recall}')
    print("___________________________________")
    return best_k


def random_forest(X_train, X_test, y_train, y_test):
    # choose_Model_with_best_feature(data)
    # data = data.drop(columns=['weight'])
    # data = data.drop(columns=['height'])
    # data = data.drop(columns=['active'])
    # # data = data.drop(columns=['age'])
    # data = data.drop(columns=['ap_lo'])
    # # data = data.drop(columns=['gluc'])
    # data = data.drop(columns=['gender_2'])
    # data = data.drop(columns=['smoke'])
    # data = data.drop(columns=['alco'])
    # data = data.drop(columns=['cholesterol'])

    model = RandomForestClassifier(criterion="gini",
                                   n_estimators=1000,
                                   max_depth=500,
                                   min_samples_split=40,
                                   random_state=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(Y_test, y_pred)
    print("___________________________________")
    print("PRECISION FOR RANDOM FOREST:", precision)
    print("___________________________________")
    print("Recall FOR RANDOM FOREST:", recall)
    print("___________________________________")
    print("ACCURACY FOR RANDOM FOREST:", accuracy)
    print("___________________________________")
    print("CONFUSION FOR RANDOM FOREST", confusion)
    print("___________________________________")
    # Print classification report and other metrics
    print("Classification Report for Random forest:")
    print(classification_report(y_test, y_pred))
    print("___________________________________")

    # Create a heatmap of the confusion matrix
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix FOR RANDOM FOREST")
    plt.show()


def random_forest_cv(X_train, y_train, cv_folds=10):
    model = RandomForestClassifier(criterion="gini",
                                   n_estimators=1000,
                                   max_depth=500,
                                   min_samples_split=40,
                                   random_state=5)

    accuracy_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
    precision_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='precision')
    recall_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='recall')

    print("Cross-Validation Scores:")
    print(f"Mean Accuracy: {accuracy_scores.mean()}")
    print(f"Mean Precision: {precision_scores.mean()}")
    print(f"Mean Recall: {recall_scores.mean()}")


def Logistic_Regression_Model(X_train, X_test, y_train, y_test):
    class_weights = {0: 1, 1: 1.3}
    # Initialize Logistic Regression model with balanced class weights
    log_reg = LogisticRegression(solver='saga', class_weight=class_weights, penalty='l2', C=1000, max_iter=3000)

    # Fit the model to the training data
    log_reg.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred_log_reg = log_reg.predict(X_test)

    # Calculate metrics
    accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
    precision_log_reg = precision_score(y_test, y_pred_log_reg)
    recall_log_reg = recall_score(y_test, y_pred_log_reg)
    f1score = f1_score(y_test, y_pred_log_reg)

    # Print classification report and other metrics
    print("Classification Report for Logistic Regression:")
    print(classification_report(y_test, y_pred_log_reg))
    print("--------------------------")
    print("Accuracy:", accuracy_log_reg)
    print("--------------------------")
    print("Precision:", precision_log_reg)
    print("--------------------------")
    print("Recall:", recall_log_reg)
    print("--------------------------")
    print(f" F1-SCORE = {f1score}")
    print("--------------------------")
    confusion = confusion_matrix(y_test, y_pred_log_reg)

    # Create a heatmap of the confusion matrix
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix FOR LOGISTIC REGRESSION")
    plt.show()

    ##########################    ANALYTICS #########################
    # Identifying and analyzing the instances where the model made errors
    errors = X_test.copy()
    errors['actual'] = y_test
    errors['predicted'] = y_pred_log_reg
    errors = errors[errors['actual'] != errors['predicted']]

    # Displaying a summary of the errors
    print(f'NUMBER OF MISS CLASSIFIED:{len(errors)}')
    # Retrieve the mean and standard deviation for the 'age' feature
    age_mean = scaler.mean_[0]
    age_std = scaler.scale_[0]
    original_average_age = (errors["age"].mean() * age_std) + age_mean
    print(f'Original average age: {original_average_age}')
    # Count the number of women (where gender_2 is 0)
    number_of_women = errors[errors['gender_2'] == 0].shape[0]
    # Calculate the total number of rows in the errors DataFrame
    total_rows = errors.shape[0]
    # Calculate the percentage of women
    percentage_of_women = (number_of_women / total_rows) * 100
    print("___________________________________")
    print(f"Percentage of women in missed classified labels: {percentage_of_women}")

    # AVERAGE WEIGHT, HEIGHT FOR MISS CLASSIFIED POINTS
    # Retrieve the mean and standard deviation for the 'height' feature
    average_error(2, 'height', errors)
    # Retrieve the mean and standard deviation for the 'ap_hi' feature
    print(f'SMOKE AVERAGE FOR MISS CLASSIFIED POINTS: {errors["smoke"].mean()}')
    print(f'GLUCOSE AVERAGE FOR MISS CLASSIFIED POINTS: {errors["gluc"].mean()}')
    print("___________________________________")
    print(f'ACTIVE AVERAGE FOR MISS CLASSIFIED POINTS: {errors["active"].mean()}')
    print("___________________________________")
    print(f'ALCOHOL AVERAGE FOR MISS CLASSIFIED POINTS: {errors["alco"].mean()}')
    print("___________________________________")
    print(f"NUMBER OF PATIENTS WHICH MARKED AS HAVE THE disease WHERE THEY WEREN'T :{errors[errors['actual'] == 1].shape[0]}")
    print(f"NUMBER OF PATIENTS WHICH MARKED AS positive THE disease WHERE THEY WEREN'T :{errors[errors['actual'] == 0].shape[0]}")



    # ERROR PLOT
    error_counts = errors['active'].value_counts()
    error_plot(error_counts, 'active')
    error_counts = errors['gluc'].value_counts()
    error_plot(error_counts, 'gluc')
    error_counts = errors['alco'].value_counts()
    error_plot(error_counts, 'alco')
    error_counts = errors['smoke'].value_counts()
    error_plot(error_counts, 'smoke')
    error_counts = errors['gender_2'].value_counts()
    error_plot(error_counts, 'gender_2')
    error_counts = errors['active'].value_counts()
    error_plot(error_counts, 'active')


def error_plot(error_counts, feature):
    sns.barplot(x=error_counts.index, y=error_counts.values)
    plt.xlabel(f'Actual Class ({feature})')
    plt.ylabel('Number of Miss classifications')
    plt.title('Distribution of Misclassified Instances per Class')
    plt.show()


def average_error(feature_num, feature_name, errors):
    mean = scaler.mean_[feature_num]
    std = scaler.scale_[feature_num]
    original_average = (errors[feature_name].mean() * std) + mean
    print("___________________________________")
    print(f'AVERAGE {feature_name} FOR MISS CLASSIFIED {feature_name}:{original_average}')
    print("___________________________________")


cardio_data = read_data_file()  # read the csv file
if cardio_data is None:
    print("No such file or directory, EXIT")
else:
    # before cleaning
    print("BEFORE CLEANING")
    data_visualization(cardio_data, 'height',desc='height')
    data_visualization(cardio_data, 'age',desc='age in days')
    data_visualization(cardio_data, 'weight',desc='weight')
    data_visualization(cardio_data, 'ap_hi', desc='ap_hi')
    data_visualization(cardio_data, 'ap_lo', desc='ap_lo')
    data_visualization(cardio_data, 'gender', desc='gender')
    data_visualization(cardio_data, 'gluc', desc='glucose')
    data_visualization(cardio_data, 'cholesterol', desc='cholesterol')
    data_visualization(cardio_data, 'smoke', desc='smoke')
    data_visualization(cardio_data, 'alco', desc='alco')
    data_visualization(cardio_data, 'active', desc='active')
    data_visualization(cardio_data, 'cardio', desc='cardio{Have the disease or not}')

    cardio_data = clean_data(cardio_data)  # preprocessing the data
    X = cardio_data.iloc[:, :-1]
    Y = cardio_data.iloc[:, -1]

    print("___________________________________")
    print("0: WOMEN, 1:MEN")
    print("___________________________________")

    # after cleaning
    print("AFTER CLEANING")
    print("___________________________________")
    data_visualization(cardio_data, 'height', desc='height')
    data_visualization(cardio_data, 'age', desc='age in days')
    data_visualization(cardio_data, 'weight', desc='weight')
    data_visualization(cardio_data, 'ap_hi', desc='ap_hi')
    data_visualization(cardio_data, 'ap_lo', desc='ap_lo')
    data_visualization(cardio_data, 'gender_2', desc='gender')
    data_visualization(cardio_data, 'gluc', desc='glucose')
    data_visualization(cardio_data, 'cholesterol', desc='cholesterol')
    data_visualization(cardio_data, 'smoke', desc='smoke')
    data_visualization(cardio_data, 'alco', desc='alco')
    data_visualization(cardio_data, 'active', desc='active')
    data_visualization(cardio_data, 'cardio', desc='cardio{Have the disease or not}')

    # split the data into training and testing
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    random_forest_cv(X_train,Y_train,cv_folds=5)
    K_NN(X_train, X_test, Y_train, Y_test, 1)
    K_NN(X_train, X_test, Y_train, Y_test, 3)
    print("Calculating the BEST K........")
    best_k = K_NN_with_cross_validation(cardio_data)
    print("-----------------------------------")
    K_NN(X_train, X_test, Y_train, Y_test, best_k)
    random_forest(X_train, X_test, Y_train, Y_test)
    Logistic_Regression_Model(X_train, X_test, Y_train, Y_test)
