# -*- coding: utf-8 -*-
"""
@author: hughy
"""
import numpy as np
import pandas as pd
import random
import plotly.express as px
import plotly.figure_factory as ff
from plotly.offline import plot
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import pycountry
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
np.random.seed(24)

def plots(df):
    Salary = df["salary_in_usd"]
    name = df["salary_in_usd"].name
    Rfig = make_subplots(rows=1, cols=2,
                       subplot_titles=('Percentages', 'Countplot'))
    
    fig1 = ff.create_distplot([Salary],[name], bin_size=5000, colors = ['#A56CC1'],show_hist=False, show_curve=True, show_rug=False)
 

    fig2 = ff.create_distplot([Salary],[name], bin_size=5000, colors = ['#A56CC1'],histnorm= '',show_rug=False)
    
    
    Rfig.add_trace(fig1.data[0], row=1, col=1)
    Rfig.add_trace(fig2.data[0], row=1, col=2)
    
    Rfig.update_layout(
        title='Distribution Plot for Salary',
        showlegend=False
    )
    Rfig.write_html("DistPlot.html")
    
    Rfig1 = px.box(df,y=df["salary_in_usd"])
    Rfig1.update_layout(
        title='Quantiles'
    )
    Rfig1.write_html("BoxPlot.html")
    

    Experience = df['experience_level'].value_counts()
    Size = df['company_size'].value_counts()  
    Year = df['work_year'].value_counts() 
    Remote = df['remote_ratio'].value_counts()
    Rfig2 = make_subplots(rows=2, cols=2, subplot_titles=("Experience Level", "Company Size", "Work Year", "Remote Ratio"), specs=[[{'type': 'domain'}, {'type': 'domain'}], [{'type': 'domain'}, {'type': 'domain'}]])    
    Rfig2.add_trace(go.Pie(labels=Experience.index, values=Experience.values), row=1, col=1)
    Rfig2.add_trace(go.Pie(labels=Size.index, values=Size.values), row=1, col=2)
    Rfig2.add_trace(go.Pie(labels=Year.index, values=Year.values), row=2, col=1)
    Rfig2.add_trace(go.Pie(labels=Remote.index, values=Remote.values), row=2, col=2)
    Rfig2.update_layout(
        title='Pie Charts',
        showlegend=False)
    Rfig2.write_html("PieCharts.html")
    
    job_titles = df['job_title'].value_counts()[:10]
    Rfig3 = px.bar(y = job_titles.values, x = job_titles.index, 
        text = job_titles.values, title = 'Popular Job titles')
    Rfig3.update_layout(xaxis_title = "Job Designations", yaxis_title = "Count")
    Rfig3.write_html("JobTitles.html")
    

    mean_salary_workyear = df.groupby('work_year')['salary_in_usd'].mean().round().reset_index()
    Rfig4 = px.bar(mean_salary_workyear, x='work_year', y='salary_in_usd',
              text='salary_in_usd', title='Mean Salary by Experience Level',
              labels={'work_year': 'Work Year', 'salary_in_usd': 'Mean Salary (USD)'},color='work_year')
    Rfig4.update_layout(
                   showlegend=False)
    Rfig4.write_html("SalaryByWorkyear.html")
        
    
    mean_salary_experience = df.groupby('experience_level')['salary_in_usd'].mean().round().reset_index()
    Rfig5 = px.bar(mean_salary_experience, x='experience_level', y='salary_in_usd',
              text='salary_in_usd', title='Mean Salary by Experience Level',
              labels={'experience_level': 'Experience Level', 'salary_in_usd': 'Mean Salary (USD)'},color='experience_level')
    Rfig5.update_layout(
                   showlegend=False)
    Rfig5.write_html("SalaryByExperience.html")
    
    mean_salary_location = df.groupby('company_location')['salary_in_usd'].mean().reset_index()
    # Create choropleth map for average salary by company location
    Rfig6 = px.choropleth(mean_salary_location, locations='company_location', locationmode='country names',
                    color='salary_in_usd', title='Average Salary by Company Location')
    Rfig6.write_html("SalaryBylocation.html")
    
def remove_outliers_zscore(data, threshold=3):
    # Compute the z-scores for each data point
    z_scores = np.abs((data - data.mean()) / data.std())

    # Filter out the data points that have a z-score greater than the threshold
    filtered_data = data[z_scores <= threshold]

    return filtered_data


def remove_outliers_iqr(data, col,multiplier=1.5):
    # Calculate the IQR (Interquartile Range)
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1

    # Define the upper and lower bounds
    # Lower bound is negative here since the salary is right-skewed
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    # Filter out the data points that fall outside the bounds
    filtered_data = data[~((data[col] < lower_bound) | (data[col] > upper_bound))]
    return filtered_data

def salary_range_models(data):
    quantiles = [0, 1/8, 1/4, 3/8, 1/2, 5/8, 3/4, 7/8, 1]
    bin_edges = [filtered_data['salary_in_usd'].quantile(q) for q in quantiles]
    salary_labels = ['1', '2', '3', '4', '5', '6', '7' , '8']
    data['salary_range'] = pd.cut(data['salary_in_usd'], bins=bin_edges, labels=salary_labels, include_lowest=True)
    y = data['salary_range']
    data = data.drop(['salary_range'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2)

    classifiers = [
        ('Logistic Regression', LogisticRegression(max_iter=1000)),
        ('Random Forest', RandomForestClassifier()),
        ('Gradient Boosting', HistGradientBoostingClassifier())
        ]
    accuracies = []
    best_model = None
    best_score = -np.inf
    for name, clf in classifiers:
        pipeline = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} - Accuracy: {accuracy:.4f}")

        if accuracy > best_score:
            best_score = accuracy
            best_model = pipeline
    """    
    BaseModel =  HistGradientBoostingClassifier()
    param_grid = { 
        'learning_rate': [0.1, 0.2, 0.3],
        'max_leaf_nodes': [20, 30,40],
        'min_samples_leaf': [20,30,40]
    }
    grid = GridSearchCV(BaseModel, param_grid, cv=5, scoring = 'accuracy')
    grid.fit(X_train, y_train)
    print('Best hyperparameters:',grid.best_params_)
    grid_predictions = grid.predict(X_test)
    score = grid.score(X_test, y_test)
    print(round(score*100,2))
    """
    
    titles_options = [
    ("Confusion matrix, without normalization", None),
    ("Normalized confusion matrix", "true"),
    ]
    for title, normalize in titles_options:
        disp = ConfusionMatrixDisplay.from_estimator(
            best_model,
            X_test,
            y_test,
            display_labels=salary_labels,
            cmap=plt.cm.Blues,
            normalize=normalize,
        )
        disp.ax_.set_title(title)
    
        print(title)
        print(disp.confusion_matrix)
    
    plt.show()
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=salary_labels))
    return 

#Load the data

data = pd.read_csv("ds_salaries.csv")

#EDA
print(data.head())
#Using 'salary_in_usd' is more suitable since the column 'salary' is measured in different currency units. Hence, dropping
data.drop(data[['salary','salary_currency']], axis = 1, inplace = True)

#Check for missing data
print(data.isnull().sum()) 
#Check for unique value
print(data.nunique())

data['experience_level'] = data['experience_level'].replace({'SE': 'Senior',
                                                             'EN': 'Entry level',
                                                             'EX': 'Executive level',
                                                             'MI': 'Mid level'})

data['employment_type'] = data['employment_type'].replace({'FL': 'Freelancer',
                                                           'CT': 'Contractor',
                                                           'FT': 'Full-time',
                                                           'PT': 'Part-time'})

data['company_size'] = data['company_size'].replace({'S': 'SMALL',
                                                     'M': 'MEDIUM',
                                                     'L': 'LARGE'})

data['remote_ratio'] = data['remote_ratio'].astype(str)
data['remote_ratio'] = data['remote_ratio'].replace({'0': 'On-Site',
                                                     '50': 'Half-Remote',
                                                     '100': 'Full-Remote'})
country_map = {}
for country in pycountry.countries:
    country_map[country.alpha_2] = country.name
# replace values in 'employee_residence' column using dictionary
data['employee_residence'] = data['employee_residence'].replace(country_map)
data['company_location'] = data['company_location'].replace(country_map)

"""
We are left with 9 Columns with 3755 entries with no missing data
The target variable is salary_in_usd
"""
plots(data)

#Building a Machine Learning Model



#Since the data is right skewed , we use the IRQ method instead of Z-Score for removing outliers
df = data.copy()
filtered_data = remove_outliers_iqr(df,'salary_in_usd')
filtered_data = filtered_data.drop(['work_year','job_title'], axis=1)
categorical_features = ['experience_level', 'employment_type', 'employee_residence', 'company_location', 'company_size', 'remote_ratio']
encoder = LabelEncoder()
for feature in categorical_features:
    filtered_data[feature] = encoder.fit_transform(filtered_data[feature])
salary_range_models(filtered_data)




