import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the dataset
df = pd.read_csv("hotel_bookings.csv")
print(f"Shape: {df.shape}")
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum().sort_values(ascending=False))

# Handle Missing Values
df = df.fillna({
    'children': 0,
    'agent': 0,
    'company': 0,
    'country': 'Unknown'
})


# Format & Feature Engineering
df['arrival_date'] = pd.to_datetime(df['arrival_date_year'].astype(str) + '-' +
                                    df['arrival_date_month'] + '-' +
                                    df['arrival_date_day_of_month'].astype(str))

df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
df['total_guests'] = df['adults'] + df['children'] + df['babies']

# Save cleaned data
df.to_csv("cleaned_hotel_bookings.csv", index=False)
