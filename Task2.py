import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("TitanicMachine Learning from Disaster/train.csv")

# Summarize missing data before cleaning
print("Missing data before cleaning:")
print(df.isnull().sum())

# Data cleaning
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df = df.drop('Cabin', axis=1)

# Convert types
df['Sex'] = df['Sex'].astype('category')
df['Embarked'] = df['Embarked'].astype('category')

# Summarize missing data after cleaning
print("\nMissing data after cleaning:")
print(df.isnull().sum())

# Basic statistics
df.describe()

# Group-based insights
print(df.groupby('Sex', observed=False)['Survived'].mean())
print(df.groupby('Pclass', observed=False)['Survived'].mean())

# Distribution of Survived
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.show()

# Distribution of Age, Fare, Pclass, Sex, Embarked
sns.histplot(df['Age'].dropna(), bins=30)
plt.title('Age Distribution')
plt.show()
sns.countplot(x='Sex', data=df)
plt.title('Gender Distribution')
plt.show()
     
# Survival rate by gender
sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Survival Rate by Gender')
plt.show()

# Survival rate by class
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Survival Rate by Passenger Class')
plt.show()

#Survival rate by embarked
sns.barplot(x='Embarked', y='Survived', data=df)
plt.title('Survival Rate by Port of Embarkation')
plt.show()

# Heatmap for correlations
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


pivot = df.pivot_table(index='Pclass', columns='Sex', values='Survived', observed=False)
sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt=".2f")
plt.title('Survival Rate by Class and Gender')
plt.show()






