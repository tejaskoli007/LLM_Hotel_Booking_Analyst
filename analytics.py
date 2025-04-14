# analytics.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create 'plots' directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Load cleaned data
df = pd.read_csv("cleaned_hotel_bookings.csv")

# -----------------------------------------------
# 1. Revenue Trends (Monthly)
# -----------------------------------------------
def revenue_trends():
    df['month_year'] = pd.to_datetime(df['arrival_date']).dt.to_period('M')
    revenue_df = df[df['is_canceled'] == 0].copy()
    revenue_df['revenue'] = revenue_df['adr'] * revenue_df['total_nights']
    monthly_revenue = revenue_df.groupby('month_year')['revenue'].sum().sort_index()

    plt.figure(figsize=(12, 6))
    monthly_revenue.plot(kind='line', marker='o')
    plt.title("Monthly Revenue Trend")
    plt.xlabel("Month-Year")
    plt.ylabel("Revenue (€)")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("plots/revenue_trends.png")
    plt.close()

# -----------------------------------------------
# 2. Cancellation Rate
# -----------------------------------------------
def cancellation_rate():
    total = len(df)
    cancelled = df['is_canceled'].sum()
    rate = (cancelled / total) * 100

    plt.figure(figsize=(5, 5))
    plt.pie([cancelled, total - cancelled], labels=['Canceled', 'Not Canceled'], autopct='%1.1f%%', colors=['#FF6B6B', '#4ECDC4'])
    plt.title(f"Cancellation Rate ({rate:.2f}%)")
    plt.savefig("plots/cancellation_rate.png")
    plt.close()

# -----------------------------------------------
# 3. Geographical Distribution of Bookings
# -----------------------------------------------
def geo_distribution():
    top_countries = df['country'].value_counts().head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_countries.values, y=top_countries.index, palette='viridis')
    plt.title("Top 10 Countries by Number of Bookings")
    plt.xlabel("Number of Bookings")
    plt.ylabel("Country")
    plt.tight_layout()
    plt.savefig("plots/geo_distribution.png")
    plt.close()

# -----------------------------------------------
# 4. Lead Time Distribution
# -----------------------------------------------
def lead_time_distribution():
    plt.figure(figsize=(10, 6))
    sns.histplot(df['lead_time'], bins=50, kde=True, color='steelblue')
    plt.title("Lead Time Distribution")
    plt.xlabel("Lead Time (days)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("plots/lead_time_distribution.png")
    plt.close()

# -----------------------------------------------
# Execute All Analytics
# -----------------------------------------------
if __name__ == "__main__":
    print("Generating analytics and saving plots...")
    revenue_trends()
    cancellation_rate()
    geo_distribution()
    lead_time_distribution()
    print("All plots saved in the 'plots/' directory.")
