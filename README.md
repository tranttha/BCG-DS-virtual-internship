# BCG-DS-virtual-internship

## Task 1: Business Understanding & Hypothesis Framing 
#### The brief from PowerCo
The Associate Director (AD) of the Data Science team held a team meeting to discuss the client brief. You’ll be working closely with Estelle Altazin, a senior data scientist on your team.

Here are the key takeaways from the meeting:

- Your client is PowerCo - a major gas and electricity utility that supplies to small and medium sized enterprises.
- The energy market has had a lot of change in recent years and there are more options than ever for customers to choose from.
- PowerCo are concerned about their customers leaving for better offers from other energy providers. When a customer leaves to use another service provider, this is called churn.
- This is becoming a big issue for PowerCo and they have engaged BCG to help diagnose the reason why their customers are churning.

During the meeting your AD discussed some potential reasons for this churn, one being how “sensitive” the price is. In other words, how much is price a factor in a customer’s choice to stay with or leave PowerCo?

So, now it’s time for you to investigate this hypothesis.

### Your task - we need to understand PowerCo’s problem in detail

First things first, you and Estelle need to understand the problem that PowerCo is facing at a deeper level and plan how you’ll tackle it. If you recall the 5 steps in the Data Science methodology, this is called “business understanding & problem framing”.

Your AD wants you and Estelle to email him by COB today outlining:

1. the data that we’ll need from the client, and
2. the techniques we’ll use to investigate the issue.


***Submission summarization***: Drafted an email outlining the steps needed to investigate customer churn for PowerCo
- Identified key factors contributing to customer churn, including price sensitivity, customer service quality, billing process ease, service reliability, and contract flexibility.
- Requested specific data from the past five years, including customer churn data, plan details, demographic data, customer service logs, and satisfaction survey data.
Proposed developing a predictive model using logistic regression to assess churn risk across different customer segments.
- Explained the methodology, including segmenting customers, training the model, and validating it through cross-validation.
- Highlighted the deliverable as a presentation of the model's results with actionable recommendations.



### Task 2: Exploratory Data Analysis

#### Exploratory data analysis
The client has sent over 2 datasets and it your responsibility to perform some exploratory data analysis.

**What is exploratory data analysis?**

Exploratory data analysis (EDA) is a technique used by a Data Scientist to gain a holistic understanding of the data that they are working with.

It is mainly based around using statistical techniques (such as descriptive statistics) and visualizations to gain a deeper understanding of the statistical properties that the data holds.


***Submission summarization***: Perfromed EDA and primary visualization including:
- Getting Descriptive statiscs of the data, including: datatypes of columns, data description of dataframes (count, unique value, value distribution, quantile range, min max), check for duplications 
- Handled null of categorical of dataset, visualized value distribution of categorical columns using bar plots, visualized churn distribution of categorical columns using stacked bar plots
- Visualized churn distribution of datetime columns using stacked area plots and grouped bar plots.
- Perform data visualization of correlation coefficents using heatmap 
- Visualized numerical distribution using histograms  

### Task 3: Feature Engineering 

***Submission summarization***: Perfromed data wrangling techniques and Feature Engineering including:
- One-hot encoded categorical features
- Inspected and Implemented scaling transformation based on skewness and kurtosis, including (Boxcox, log10, yeo-johnson )
- Engineered feautures based on datetime features, numerical features and combination of features 
- Primary model for testing purposes based on cleaned dataframe, evaluated engineered feature set, cleaned feature set scored : 90% however model had low F1 score of 0.09 and low recall of 0.05 due to the imbalance nature of the target class. (churned value of 1 is 10% of total rows), implemented SMOTE data sampling technique to test out 

### Task 4: Modeling 
***Submission summarization***: Performed traditional 

