#!/usr/bin/env python
# coding: utf-8

# # FIFA Player Dataset Analysis by ARITRA NATH (1NT16CS144)

# Data Science Assignment for LA-1 component.

# ![title](header.jpg)
# 
# Dataset used: - 
# 
# "data.csv" (https://www.kaggle.com/karangadiya/fifa19)
# This dataset includes the latest edition FIFA 2019 players attributes like Age, Nationality, Overall, Potential, Club, Value, Wage, Preferred Foot, International Reputation, Weak Foot, Skill Moves, Work Rate, Position, Jersey Number, Joined, Loaned From, Contract Valid Until, Height, Weight, LS, ST, RS, LW, LF, CF, RF, RW, LAM, CAM, RAM, LM, LCM, CM, RCM, RM, LWB, LDM, CDM, RDM, RWB, LB, LCB, CB, RCB, RB, Crossing, Finishing, Heading, Accuracy, ShortPassing, Volleys, Dribbling, Curve, FKAccuracy, LongPassing, BallControl, Acceleration, SprintSpeed, Agility, Reactions, Balance, ShotPower, Jumping, Stamina, Strength, LongShots, Aggression, Interceptions, Positioning, Vision, Penalties, Composure, Marking, StandingTackle, SlidingTackle, GKDiving, GKHandling, GKKicking, GKPositioning, GKReflexes, and Release Clause.

# In[1]:


# Importing necessary libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


my_data=pd.read_csv('data.csv',index_col=0) # importing the dataset


# In[3]:


my_data.head(10) # Displaying the first ten rows of the dataset


# In[4]:


my_data.tail(10) # Displaying the last 10 rows of the dataset


# In[5]:


my_data.info() # Gathering information about the DataFrame


# In[6]:


my_data.describe() # Gathering statistical information regarding the dataset


# In[7]:


#Checking the shape of data
my_data.shape


# i.e. We have a total of 18207 rows and 88 columns of data to work with.
# 
# # Seaborn.lmplot (Linear Model Plot)
# 
# seaborn.lmplot(x, y, data, hue=None, col=None, row=None, palette=None, col_wrap=None, height=5, aspect=1, markers='o', sharex=True, sharey=True, hue_order=None, col_order=None, row_order=None, legend=True, legend_out=True, x_estimator=None, x_bins=None, x_ci='ci', scatter=True, fit_reg=True, ci=95, n_boot=1000, units=None, order=1, logistic=False, lowess=False, robust=False, logx=False, x_partial=None, y_partial=None, truncate=False, x_jitter=None, y_jitter=None, scatter_kws=None, line_kws=None, size=None)
# 
# Plot data and regression model fits across a FacetGrid.
# 
# This function combines regplot() and FacetGrid. It is intended as a convenient interface to fit regression models across conditional subsets of a dataset.
# 
# When thinking about how to assign variables to different facets, a general rule is that it makes sense to use hue for the most important comparison, followed by col and row. However, always think about your particular dataset and the goals of the visualization you are creating.
# 
# There are a number of mutually exclusive options for estimating the regression model. See the tutorial for more information.
# 
# The parameters to this function span most of the options in FacetGrid, although there may be occasional cases where you will want to use that class and regplot() directly.
# 
# # A Linear Model plot for Age vs Potential

# In[8]:


#Relationship between 'Age' and 'Potential'.

sns.set_style('whitegrid')
sns.lmplot(x='Age', y="Potential", data=my_data, 
           aspect=2)


# Hence, we can deduce that the potential of a player reduces with increasing age.
# 
# # Seaborn.countplot
# 
# seaborn.countplot(x=None, y=None, hue=None, data=None, order=None, hue_order=None, orient=None, color=None, palette=None, saturation=0.75, dodge=True, ax=None, **kwargs)
# Show the counts of observations in each categorical bin using bars.
# 
# A count plot can be thought of as a histogram across a categorical, instead of quantitative, variable. The basic API and options are identical to those for barplot(), so you can compare counts across nested variables.
# 
# Input data can be passed in a variety of formats, including:
# 
# Vectors of data represented as lists, numpy arrays, or pandas Series objects passed directly to the x, y, and/or hue parameters.
# A “long-form” DataFrame, in which case the x, y, and hue variables will determine how the data are plotted.
# A “wide-form” DataFrame, such that each numeric column will be plotted.
# An array or list of vectors.
# In most cases, it is possible to use numpy or Python objects, but pandas objects are preferable because the associated names will be used to annotate the axes. Additionally, you can use Categorical types for the grouping variables to control the order of plot elements.
# 
# This function always treats one of the variables as categorical and draws data at ordinal positions (0, 1, … n) on the relevant axis, even when the data has a numeric or date type.
# 
# # countplot for analysing the Age Group of players

# In[9]:


plt.figure(figsize=(15,8))
plt.title("Age vs Count")
sns.countplot(x ='Age',data=my_data, palette='hls')


# Hence, we can deduce that the maximum number of players participating in FIFA lie in the age group of 21-26.
# 
# # countplot for analysing foot preference of the players

# In[10]:


plt.figure(figsize=(10,8))
plt.title("Player Foot Preference")
sns.countplot(x ='Preferred Foot',data=my_data, palette='coolwarm')


# Therefore, it is much evident from the above graph that about 14000 players prefer their right foot while around 4000 of them use their left foot for playing.
# 
# # countplot to visualise Player Positions

# In[11]:


#Visualising all 'Position'.

plt.figure(figsize=(15,8))
plt.title("Preferred Player Positions")
sns.set_style('whitegrid')
sns.countplot(x='Position',data=my_data, palette='coolwarm')


# The above graph clearly tells us that the most preferred position by the players is that of the STRIKER.
# 
# # line plot to visualise the No of players playing for each of the top 30 countries.

# In[12]:


#Visualisation of the main 'nationalities'.

nat=pd.value_counts(my_data['Nationality'])
def f(t):
    return t * t
nat=nat.head(30)
sns.set_style('whitegrid')
plt.figure(figsize=(15,8))
plt.title('No. of players per nation for the top 30 nations')
plt.plot(nat)
plt.xticks(rotation=90)
plt.xlabel("Nations")
plt.ylabel('No. of Players')


# From the above graph we can figure out that the maximum number of players playing in the tournament originate from ENGLAND.
# 
# # seaborn.kdeplot (Kernel Density Estimate Plot)
# 
# seaborn.kdeplot(data, data2=None, shade=False, vertical=False, kernel='gau', bw='scott', gridsize=100, cut=3, clip=None, legend=True, cumulative=False, shade_lowest=True, cbar=False, cbar_ax=None, cbar_kws=None, ax=None, **kwargs)
# 
# In statistics, kernel density estimation (KDE) is a non-parametric way to estimate the probability density function of a random variable. Kernel density estimation is a fundamental data smoothing problem where inferences about the population are made, based on a finite data sample.
# 
# # kdeplot to estimate the probability density of the nationality of players

# In[13]:


sns.kdeplot(nat, shade=True)


# # seaborn.jointplot
# 
# seaborn.jointplot(x, y, data=None, kind='scatter', stat_func=None, color=None, height=6, ratio=5, space=0.2, dropna=True, xlim=None, ylim=None, joint_kws=None, marginal_kws=None, annot_kws=None, **kwargs)
# 
# # A jointplot to show the relationship between the Age and Agility of the players

# In[14]:


#Relationship between 'Age' and 'Agility'.

sns.set(style='ticks', color_codes=True)
g = sns.JointGrid(x='Age', y='Agility',data=my_data, height=8)
#g = g.plot(sns.regplot, sns.distplot)
g = g.plot_joint(sns.kdeplot, color="r", shade=True)
g = g.plot_marginals(sns.kdeplot, color="r", shade=True)


# The darker regions in the above graph determine that the frequency of the players in that region is the maximum.
# It is also evident that the resulting jointplot is the intersection of the 'X' and 'Y' axis plots of the histograms.

# # Line plot to analyse the Average Rating by Age

# In[15]:


df_p = my_data.groupby(['Age'])['Potential'].mean()
df_o = my_data.groupby(['Age'])['Overall'].mean()

df_summary = pd.concat([df_p, df_o], axis=1)
plt.figure(figsize=(15,8))
ax = df_summary.plot()
ax.set_ylabel('Rating')
ax.set_title('Average Rating by Age')


# In[16]:


# Creating a list containing only the required column names for further analysis.

new_columns=['Name','Wage','Value','Age','Overall','Potential','Special','Release Clause',
        'International Reputation','Weak Foot','Skill Moves','Crossing',
       'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
       'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
       'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
       'GKKicking', 'GKPositioning', 'GKReflexes']


# In[17]:


#New dataframe

df2 = pd.DataFrame(my_data,columns=new_columns)

df2.head(20) # Displaying the first 20 rows of the new dataframe. 


# In[18]:


#Checking missing data
df2.isnull().sum()


# We can see above that a lot of our data contains NULL values.
# These null values need to be filled with average data before proceeding.

# In[19]:


#Fixing missing data
for column in df2.iloc[:,8:]:
    df2[column].fillna(value = df2[column].mean(),inplace=True)


# Slicing the '€' symbols from the 'Wage' and 'Value' fields and converting it to a numerical value for post-processing.

# In[20]:


#coverting non-numerical data into numerical
def convert(item):
    if 'M' in item:
        item=float(item.split('€')[1].split('M')[0])*1000000
    elif 'K' in item:
        item=float(item.split('€')[1].split('K')[0])*1000
    else: 
        item=float(item.split('€')[1])
#     item=np.float64(item)
    return item

df2['Value'] = df2['Value'].apply(convert)
df2['Wage'] = df2['Wage'].apply(convert)

RC_mean=df2['Release Clause'].dropna().apply(convert).mean()
RC_mean="€"+str(RC_mean)
df2['Release Clause']=df2['Release Clause'].fillna(value=RC_mean).apply(convert)


# In[21]:


#Checking missing data again
df2.isnull().sum()


# It seems we don't have any null values anymore.

# In[22]:


df2.head() # Displaying the first 5 rows of the new dataset again.


# # seaborn.heatmap
# seaborn.heatmap(data, vmin=None, vmax=None, cmap=None, center=None, robust=False, annot=None, fmt='.2g', annot_kws=None, linewidths=0, linecolor='white', cbar=True, cbar_kws=None, cbar_ax=None, square=False, xticklabels='auto', yticklabels='auto', mask=None, ax=None, **kwargs)
# 
# A heatmap plots rectangular data as a color-encoded matrix.
# 
# Let's create a correlation matrix of the important player attributes and plot it into a heatmap.

# In[23]:


#Correlation matrix of part of the columns
corr_columns = ['Value','Age','Overall','Potential','Special','Release Clause',
        'International Reputation','Weak Foot','Skill Moves','Crossing',
       'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
       'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
       'Marking', 'StandingTackle', 'SlidingTackle']

#Plotting the heat-map
plt.figure(figsize=(20,15))
heat_map = sns.heatmap(df2[corr_columns].corr(),annot = True, linewidths=0.1, cmap='magma')
heat_map.set_title(label='Heatmap for player attributes', fontsize=20)
heat_map


# Correlation gives the relation between two numerical attributes in the range of 0-1.
# # seaborn.pairplot
# seaborn.pairplot(data, hue=None, hue_order=None, palette=None, vars=None, x_vars=None, y_vars=None, kind='scatter', diag_kind='auto', markers=None, height=2.5, aspect=1, dropna=True, plot_kws=None, diag_kws=None, grid_kws=None, size=None)
# 
# It plots pairwise relationships in a dataset.
# 
# By default, this function will create a grid of Axes such that each variable in data will by shared in the y-axis across a single row and in the x-axis across a single column. The diagonal Axes are treated differently, drawing a plot to show the univariate distribution of the data for the variable in that column.
# 
# It is also possible to show a subset of variables or plot different variables on the rows and columns.
# 
# This is a high-level interface for PairGrid that is intended to make it easy to draw a few common styles. You should use PairGrid directly if you need more flexibility.
# # Pairplots between 5 numerical attributes

# In[24]:


#Pairplot
sns.pairplot(df2.iloc[:,1:6]) # All rows but columns 2-7


# # Analysing the Wage and Age Distribution of top 20 players

# In[25]:


x =list(range(20))
total_width, n = 0.8, 2
width = total_width / n

plt.figure(1 , figsize = (15 , 8))
plt.xlabel('Player\'s Name')
plt.ylabel('Wage = Number * 10000, Age')
plt.title('Top 20 players Wage and Age distribution')
plt.xticks(rotation=80)
plt.bar(x,df2['Wage'].head(20)/10000, width=width, label='Wage',tick_label = df2['Name'].head(20),fc = 'y')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, df2['Age'].head(20), width=width, label='Age',fc = 'r')
plt.legend()


# The above graph is self explainatory here.
# 
# Now let's calculate the total goals scored for each club and insert the obtained data into a new dataframe.

# In[26]:


total_goals = my_data.groupby(['Club']).sum().reset_index()
club_tot = total_goals.head(35).iloc[:,[0,3]]
club_tot = club_tot.rename(columns={'Overall':'Total Goals'})
club_tot


# # seaborn.barplot
# seaborn.barplot(x=None, y=None, hue=None, data=None, order=None, hue_order=None, estimator=<function mean>, ci=95, n_boot=1000, units=None, orient=None, color=None, palette=None, saturation=0.75, errcolor='.26', errwidth=None, capsize=None, dodge=True, ax=None, **kwargs)
# 
# Shows point estimates and confidence intervals as rectangular bars.
# 
# A bar plot represents an estimate of central tendency for a numeric variable with the height of each rectangle and provides some indication of the uncertainty around that estimate using error bars. Bar plots include 0 in the quantitative axis range, and they are a good choice when 0 is a meaningful value for the quantitative variable, and you want to make comparisons against it.
# 
# # Analysing the total no. of goals scored by the top clubs

# In[27]:


plt.figure(figsize=(15,8))
sns.set(style="whitegrid")
plt.xticks(rotation=80)
sns.barplot(x='Club', y='Total Goals', data=club_tot).set_title(label="Total number of goals for the top 35 clubs",
                                                            fontsize=20)


# The above graph justifies that out of the first 35 clubs, the club 'AS Monaco' scored the highest number of goals.

# In[28]:


my_data['Club'].replace('', 'NA', inplace=True) # replacing the null values in the 'Club' column with NA


# 
# Creating a custom 10 Club array for analysing their performance.
# 

# In[29]:


club = ['Atlético Madrid', 'Chelsea', 'FC Barcelona', 'FC Bayern München', 'Juventus', 'Manchester City',
        'Manchester United', 'Paris Saint-Germain', 'Real Madrid', 'Tottenham Hotspur']


# Now filtering the data frame and saving only those rows where the 'Club' field contains either of a value present in the club array. We drop all the other rows which do not satisfy the condition. 

# In[30]:


club_data = my_data[my_data['Club'].isin(club)]
club_data = club_data.rename(columns={'Overall':'Total Goals'})
club_data


# # barplot to analyse the total goals scored by these 10 clubs

# In[31]:


plt.figure(figsize=(15,8))
sns.set(style="whitegrid")
plt.xticks(rotation=60)
sns.barplot(x='Club', y='Total Goals', data=club_data).set_title(label="Total number of goals scored by the 10 clubs"
                                                             , fontsize=20)


# Out of the above 10 clubs, it seems that Juventus is the best performer with around 80+ goals.
# 
# # Invoking the pandas.crosstab method to compute a simple crosstabulation between the attributes 'Club' & 'Age'
# 
# pandas.crosstab(index, columns, values=None, rownames=None, colnames=None, aggfunc=None, margins=False, margins_name='All', dropna=True, normalize=False)

# In[32]:


pd.crosstab(club_data.Club, club_data.Age)


# # seaborn.violinplot
# seaborn.violinplot(x=None, y=None, hue=None, data=None, order=None, hue_order=None, bw='scott', cut=2, scale='area', scale_hue=True, gridsize=100, width=0.8, inner='box', split=False, dodge=True, orient=None, linewidth=None, color=None, palette=None, saturation=0.75, ax=None, **kwargs)
# 
# Draws a combination of boxplot and kernel density estimate.
# 
# A violin plot plays a similar role as a box and whisker plot. It shows the distribution of quantitative data across several levels of one (or more) categorical variables such that those distributions can be compared. Unlike a box plot, in which all of the plot components correspond to actual datapoints, the violin plot features a kernel density estimation of the underlying distribution.

# In[33]:


plt.figure(figsize=(15,8))
plt.xticks(rotation=60)
sns.violinplot(x="Club", y="Age", data=club_data).set_title(label="Violin Plot for Club vs Age", fontsize=20)


# From the above violin-plot we can conclude the following -
# 
# - The club Atletico Madrid has the most no. of players aged around 22.
# - The club Paris Saint-Gemnain has players having the age range between 12-47.
# - Juventus has maximum players aged around 25.
# 

# # seaborn.boxplot
# seaborn.boxplot(x=None, y=None, hue=None, data=None, order=None, hue_order=None, orient=None, color=None, palette=None, saturation=0.75, width=0.8, dodge=True, fliersize=5, linewidth=None, whis=1.5, notch=False, ax=None, **kwargs)
# 
# Draws a box plot to show distributions with respect to categories.
# 
# A box plot (or box-and-whisker plot) shows the distribution of quantitative data in a way that facilitates comparisons between variables or across levels of a categorical variable. The box shows the quartiles of the dataset while the whiskers extend to show the rest of the distribution, except for points that are determined to be “outliers” using a method that is a function of the inter-quartile range.

# In[34]:


plt.figure(figsize=(15,8))
plt.xticks(rotation=60)
sns.boxplot(x="Club", y="Age", data=club_data).set_title(label="Box Plot for Club vs Age", fontsize=20)


# The above boxplot is similar to the previous violin-plot as shown in the previous analysis.
# Same results can b predicted.
