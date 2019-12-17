# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 16:21:07 2019

@author: joaop
"""

# df = pd.read_csv(r'C:\Users\joaop\OneDrive\Desktop\DM - Project\A2Z Insurance.csv')

df.set_index("Customer Identity", inplace=True)

df.columns

df.drop('Brithday Year', axis=1, inplace=True)

df.columns

################################# OUTLIERS ###################################

sns.boxplot(data=df[['Premiums in LOB: Motor', # data=df_new
            'Premiums in LOB: Household', 'Premiums in LOB: Health',
            'Premiums in LOB:  Life', 'Premiums in LOB: Work Compensations']],
           orient='h')

plt.show()



# Detecting the outliers:
outliers = pd.DataFrame()

outliers = df.loc[(df['Premiums in LOB: Motor'] > 3000) | (df['Premiums in LOB: Household'] > 3000) | 
        (df['Premiums in LOB: Health'] > 1000) | (df['Premiums in LOB: Work Compensations'] > 900) |
        (df['First Policy´s Year'] > 2016) | (df['Gross Monthly Salary'] > 6000) | (df['Customer Monetary Value'] > 2500) |
        (df['Customer Monetary Value'] < -1500) | (df['Claims Rate'] > 40), :]

# Removing the outliers:
df_new = df.merge(outliers, how='left', indicator=True)
df_new = df_new[df_new['_merge'] == 'left_only']
df_new.drop('_merge', axis=1, inplace=True)

#########################################################################################

customer = df[['First Policy´s Year', 'Educational Degree', 'Gross Monthly Salary',
               'Geographic Living Area', 'Has Children (Y=1)',
               'Customer Monetary Value', 'Claims Rate']]

consumption = df[['Premiums in LOB: Motor', 'Premiums in LOB: Household', 'Premiums in LOB: Health',
                  'Premiums in LOB:  Life', 'Premiums in LOB: Work Compensations']]