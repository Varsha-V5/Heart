import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, accuracy_score
pd.set_option("display.max_columns", None)


def import_data():
    data = pd.read_csv("Heart_Disease_Prediction.csv")
    print("Printing the dataset:\n", data.head(10))
    df = pd.DataFrame(data)
    print("Converting into DataFrame:\n", df.head(10))
    return df


def sanity_check(data):
    print("Shape:\n", data.shape)
    print("Dimension:\n", data.ndim)
    print("Describe:\n", data.describe())
    print("Info:\n", data.info())
    print("Unique:\n", data.nunique())
    print("Missing value:\n", data.isnull().sum())
    print("Duplicates:\n", data.duplicated().sum())
    return data


def out_treat(data):
    x = data[["BP", "Cholesterol", "Max HR", "ST depression"]]
    for col in x.columns:
        plt.figure(figsize=(10, 8))
        sns.boxplot(x=x[col], vert=False)
        plt.show()
    lb = LabelEncoder()
    data["Heart Disease"] = lb.fit_transform(data["Heart Disease"])
    return data


def pre_visual(data):
    plt.figure(figsize=(20, 10))
    x = data[["Age", "Sex", "Chest pain type", "BP", "Cholesterol", "FBS over 120", "EKG results",
              "Max HR", "Exercise angina", "ST depression", "Slope of ST", "Number of vessels fluro",
              "Thallium"]]
    y = data["Heart Disease"]
    for col in x.columns:
        sns.histplot(data=data, x=x[col], hue=y, multiple="stack", bins=10, kde=True)
        plt.show()
    cor = data.corr()
    sns.heatmap(cor, annot=True, cmap="coolwarm")
    plt.show()
    return data


def split_dataset(data):
    x = data[["Age", "Sex", "Chest pain type", "BP", "Cholesterol", "FBS over 120", "EKG results",
              "Max HR", "Exercise angina", "ST depression", "Slope of ST", "Number of vessels fluro",
              "Thallium"]]
    y = data["Heart Disease"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return x, y, x_train, x_test, y_train, y_test


def scale_balance(x_train, x_test, y_train):
    smote = SMOTE(random_state=42)
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
    std = StandardScaler()
    x_train_scaled = std.fit_transform(x_train_resampled)
    x_test_scaled = std.transform(x_test)
    return x_train_scaled, x_test_scaled, y_train_resampled


def model(x_train_scaled, x_test_scaled, y_train_resampled):
    grid = SVC()
    grid.fit(x_train_scaled, y_train_resampled)
    y_pred_train = grid.predict(x_train_scaled)
    y_pred_test = grid.predict(x_test_scaled)
    return y_pred_train, y_pred_test


def evaluation(y_train_resampled, y_test, y_pred_train, y_pred_test):
    acc_train = accuracy_score(y_train_resampled, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    cls_train = classification_report(y_train_resampled, y_pred_train)
    cls_test = classification_report(y_test, y_pred_test)
    y_test_df = pd.DataFrame(y_test)
    y_pred_test_df = pd.DataFrame(y_pred_test, index=y_test_df.index)
    final_df = pd.concat([y_test_df, y_pred_test_df], axis=1)
    final_df.to_csv("Y_test_and_prediction [Heart Disease Prediction].csv")
    return acc_train, acc_test, cls_train, cls_test


def post_visual(y_test, y_pred_test):
    plt.figure(figsize=(12, 8))
    cm = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(pd.DataFrame(cm), annot=True)
    plt.savefig("Confusion_matrix_Diabetes.png")
    plt.show()
    fpr, tpr, threshold = roc_curve(y_test, y_pred_test)
    plt.plot(fpr, tpr, color="blue")
    plt.savefig("ROC_Curve_Diabetes.png")
    plt.show()
    return 0


def main():
    data = import_data()
    san = sanity_check(data)
    outliers = out_treat(san)
    pre_vis = pre_visual(outliers)
    x, y, x_train, x_test, y_train, y_test = split_dataset(outliers)

    x_train_scaled, x_test_scaled, y_train_resampled = scale_balance(x_train, x_test, y_train)

    y_pred_train, y_pred_test = model(x_train_scaled, x_test_scaled, y_train_resampled)
    acc_train, acc_test, cls_train, cls_test = evaluation(y_train_resampled, y_test, y_pred_train, y_pred_test)
    print("acc train:\n", acc_train)
    print("acc test:\n", acc_test)
    print("cls train:\n", cls_train)
    print("cls test:\n", cls_test)

    visual = post_visual(y_test, y_pred_test)


if __name__ == "__main__":
    main()
