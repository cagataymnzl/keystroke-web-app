import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, \
    f1_score

import matplotlib
matplotlib.use('agg')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def decision_tree_model(data_path, test_size):
    # Load the dataset
    df = pd.read_csv(data_path)
    X = df.drop('label', axis=1)  # Adjust column names as per your dataset
    y = df['label']  # Adjust the target variable name as per your dataset

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # Initialize and train the decision tree classifier
    decision_tree = tree.DecisionTreeClassifier()
    decision_tree.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = decision_tree.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')
    conf_matrix = confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions)

    # Calculating feature importance
    importance = decision_tree.feature_importances_
    feature_importance = dict(zip(X.columns, importance))
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    features, importance = zip(*sorted_importance)

    # Plotting the feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importance)), importance, align='center')
    plt.yticks(range(len(features)), features)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance in Decision Tree Model')
    image_path = os.path.join('static', "feature_importance.png")
    plt.savefig(image_path)
    plt.close()

    # Return the results
    return accuracy, precision, recall, f1, conf_matrix, class_report, image_path
def random_forest_model(data_path, test_size):
    # Load the dataset
    df = pd.read_csv(data_path)
    X = df.drop('label', axis=1)  # Adjust column names as per your dataset
    y = df['label']  # Adjust the target variable name as per your dataset

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # Initialize and train the Random Forest classifier
    random_forest = RandomForestClassifier()
    random_forest.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = random_forest.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')

    conf_matrix = confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions)

    importance = random_forest.feature_importances_
    feature_importance = dict(zip(X.columns, importance))
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    features, importance = zip(*sorted_importance)

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importance)), importance, align='center')
    plt.yticks(range(len(features)), features)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance in Random Forest Model')
    image_path = os.path.join('static', "feature_importance.png")
    plt.savefig(image_path)
    plt.close()

    # Return the results
    return accuracy, precision, recall, f1, conf_matrix, class_report, image_path

def gaussian_naive_bayes_model(data_path, test_size):
    # Load the dataset
    df = pd.read_csv(data_path)
    X = df.drop('label', axis=1)
    y = df['label']

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # Initialize and train the Gaussian Naive Bayes classifier
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = gnb.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')

    conf_matrix = confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions)

    # Return the results
    return accuracy, precision, recall, f1, conf_matrix, class_report, ""

def gradient_boosting_model(data_path, test_size):
    # Load the dataset
    df = pd.read_csv(data_path)
    X = df.drop('label', axis=1)  # Adjust column names as per your dataset
    y = df['label']  # Adjust the target variable name as per your dataset

    # Encode categorical labels if needed
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # Initialize and train the Gradient Boosting Classifier
    gbm = GradientBoostingClassifier()
    gbm.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = gbm.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')
    conf_matrix = confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions)

    importance = gbm.feature_importances_
    feature_importance = dict(zip(X.columns, importance))
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    features, importance = zip(*sorted_importance)

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importance)), importance, align='center')
    plt.yticks(range(len(features)), features)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance in Gradient Boosting Machine Model')
    image_path = os.path.join('static', "feature_importance.png")
    plt.savefig(image_path)
    plt.close()
    # Return the results
    return accuracy, precision, recall, f1, conf_matrix, class_report, image_path

def plot_confusion_matrix(cm, classes, filename='confusion_matrix.png'):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    image_path = os.path.join('static', filename)
    plt.savefig(image_path)
    plt.close()
    return filename


def plot_all_metrics(models_metrics, model_names, filename='all_metrics_comparison.png'):
    df = pd.DataFrame(models_metrics, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1'])
    df_melted = df.melt(id_vars='Model', var_name='Metrics', value_name='Values')
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Metrics', y='Values', hue='Model', data=df_melted)
    plt.title('Model Comparison - All Metrics')
    plt.ylabel('Score')
    plt.xlabel('Metric')
    plt.savefig(os.path.join('static', filename))
    plt.close()
    return filename


MODEL_NAMES = {
    'decision_tree': 'Decision Tree',
    'gaussian_nb': 'Gaussian Naive Bayes',
    'random_forest': 'Random Forest',
    'gbm': 'Gradient Boosting Machine',
    'all': 'All Models'
}


@app.route('/')
def upload_file():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    cm_image_url = None
    feature_importance_url = None

    if 'file' not in request.files:
        return render_template('error.html', error="No file part in the request")

    file = request.files['file']
    model_type = request.form.get('model', 'decision_tree')

    if file.filename == '':
        return render_template('error.html', error="No selected file.")

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        train_size_percent = request.form.get('train_size', type=int)
        test_size = 1 - (train_size_percent / 100)

        try:
            df = pd.read_csv(file_path)
            classes = df['label'].unique().tolist()
            model_full_name = MODEL_NAMES.get(model_type, "Unknown Model")

            if model_type == 'all':
                models_metrics = []
                models = [random_forest_model, gaussian_naive_bayes_model, decision_tree_model, gradient_boosting_model]
                model_names = ['Random Forest', 'Gaussian Naive Bayes', 'Decision Tree', 'Gradient Boosting']

                for model, name in zip(models, model_names):
                    try:
                        acc, prec, rec, f1, _, _, _ = model(file_path, test_size)
                        models_metrics.append([name, acc, prec, rec, f1])
                    except Exception as e:
                        print(f"Error processing model {name}: {e}")

                all_metrics_chart_url = url_for('static', filename=plot_all_metrics(models_metrics, model_names))

                return render_template('all-metrics.html', all_metrics_chart_url=all_metrics_chart_url)

            else:
                model_function = {
                    'random_forest': random_forest_model,
                    'gaussian_nb': gaussian_naive_bayes_model,
                    'decision_tree': decision_tree_model,
                    'gbm': gradient_boosting_model
                }.get(model_type, decision_tree_model)

                # Get results from model function
                results = model_function(file_path, test_size)
                accuracy, precision, recall, f1, conf_matrix, class_report = results[:6]
                if len(results) > 6 and model_type in ['random_forest', 'decision_tree', 'gbm']:
                    feature_importance_plot = results[6]
                    feature_importance_url = url_for('static', filename=os.path.basename(feature_importance_plot))

                if conf_matrix is not None:
                    cm_filename = plot_confusion_matrix(conf_matrix, classes)
                    cm_image_url = url_for('static', filename=cm_filename)

                return render_template('results.html',
                                       model_used=model_full_name,
                                       accuracy=accuracy,
                                       precision=precision,
                                       recall=recall,
                                       f1=f1,
                                       cm_image_url=cm_image_url,
                                       classification_report=class_report if class_report else "No Classification Report Available",
                                       feature_importance_url=feature_importance_url)

        except Exception as e:
            os.remove(file_path)
            return render_template('error.html', error=str(e))


if __name__ == '__main__':
    app.run(debug=True)
