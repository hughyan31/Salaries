# -*- coding: utf-8 -*-
"""
@author: hughy
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from plotly.offline import plot
import plotly.io as pio
from plotly.subplots import make_subplots


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

"""
We are left with 9 Columns with 3755 entries with no missing data
The target variable is salary_in_usd
"""


Salary = data["salary_in_usd"]
count_values, bin_edges = np.histogram( data["salary_in_usd"])



fig = make_subplots(rows=1, cols=2,
                    subplot_titles=('Countplot', 'Percentage'))

fig1 = ff.create_distplot([Salary],['Salary'], bin_size=5000, colors = ['#A56CC1'],show_hist=False, show_curve=True, show_rug=False)
fig1.update_layout(
    title='Distribution Plot for Salary in distribution',
    xaxis=dict(title='Salary in USD'),
    yaxis=dict(
        visible=False
        )
)
fig1.write_html("distplotinD.html")



fig2 = ff.create_distplot([Salary],['Salary'], bin_size=5000, colors = ['#A56CC1'],histnorm= '',show_rug=False)
fig2.update_layout(
    title='Distribution Plot for Salary in count',
    xaxis=dict(title='Salary in USD'),
    yaxis=dict(title='Count')
)
fig2.write_html("distplotinC.html")


fig3 = px.box(data,y=data["salary_in_usd"])
fig2.update_layout(
    title='Quantiles'
)
fig3.write_html("BoxPlot.html")





