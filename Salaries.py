# -*- coding: utf-8 -*-
"""
@author: hughy
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from plotly.offline import plot
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import pycountry
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV



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

#Load the data
np.random.seed(77)
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



