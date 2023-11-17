import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the dataset
df = pd.read_csv('Assets/dataset.csv')

# Ensure the 'year' column is in datetime format
df['year'] = pd.to_datetime(df['year'], format='%Y')

# Set the 'year' column as the DataFrame index
df.set_index('year', inplace=True)

# Sort the DataFrame by the index (year)
df.sort_index(inplace=True)

# Take user input for the country of interest
countrySet = False

while not countrySet:
    user_country = input("Enter the name of the country for analysis: ")
    if user_country not in df['country'].unique():
        print('Invalid country name. Please try again.')
    else:
        countrySet = True

# Filter the DataFrame for the user-selected country
country_df = df[df['country'] == user_country]

# Energy Mix Trends
energy_sources = ['coal', 'gas', 'renewables', 'nuclear']
plt.figure(figsize=(14, 8))
for source in energy_sources:
    plt.plot(country_df.index, country_df[f'{source}_share_energy'], label=source.capitalize())
plt.title(f'Energy Mix Trends for {user_country} Over the Years')
plt.xlabel('Year')
plt.ylabel('Share of Energy')
plt.legend()
plt.grid(True)
plt.show()

# Per Capita Consumption
plt.figure(figsize=(12, 6))
plt.plot(country_df.index, country_df['per_capita_electricity'], marker='o', label='Per Capita Consumption')
plt.title(f'Per Capita Energy Consumption Trends for {user_country}')
plt.xlabel('Year')
plt.ylabel('Per Capita Electricity Consumption')
plt.legend()
plt.grid(True)
plt.show()

# Renewable Energy Growth
renewable_sources = ['solar', 'wind', 'hydro']
plt.figure(figsize=(14, 8))
for source in renewable_sources:
    plt.plot(country_df.index, country_df[f'{source}_share_energy'], label=source.capitalize())
plt.title(f'Renewable Energy Growth for {user_country} Over the Years')
plt.xlabel('Year')
plt.ylabel('Share of Energy')
plt.legend()
plt.grid(True)
plt.show()

# Carbon Intensity
plt.figure(figsize=(12, 6))
plt.plot(country_df.index, country_df['carbon_intensity_elec'], marker='o', color='red', label='Carbon Intensity')
plt.title(f'Carbon Intensity of Electricity Generation for {user_country} Over the Years')
plt.xlabel('Year')
plt.ylabel('Carbon Intensity')
plt.legend()
plt.grid(True)
plt.show()

# Energy and GDP Relationship
plt.figure(figsize=(12, 6))
plt.scatter(country_df['gdp'], country_df['primary_energy_consumption'], alpha=0.7)
plt.title(f'Energy Consumption vs. GDP for {user_country}')
plt.xlabel('GDP')
plt.ylabel('Primary Energy Consumption')
plt.grid(True)
plt.show()

# Fossil Fuel Consumption
fossil_fuels = ['coal', 'oil', 'gas']
plt.figure(figsize=(14, 8))
for fuel in fossil_fuels:
    plt.plot(country_df.index, country_df[f'{fuel}_consumption'], label=fuel.capitalize())
plt.title(f'Fossil Fuel Consumption Trends for {user_country} Over the Years')
plt.xlabel('Year')
plt.ylabel('Fossil Fuel Consumption')
plt.legend()
plt.grid(True)
plt.show()

# Electricity Generation and Consumption
plt.figure(figsize=(14, 8))
plt.plot(country_df.index, country_df['electricity_generation'], label='Electricity Generation', marker='o')
plt.plot(country_df.index, country_df['energy_cons_change_twh'], label='Energy Consumption', marker='o')
plt.title(f'Electricity Generation and Consumption Trends for {user_country} Over the Years')
plt.xlabel('Year')
plt.ylabel('TWH (Terawatt-hours)')
plt.legend()
plt.grid(True)
plt.show()

# Future Energy Consumption Prediction
# Filter the DataFrame for the user-selected country
country_df = df[df['country'] == user_country]

# Remove rows with NaN values in the target variable
country_df = country_df.dropna(subset=['primary_energy_consumption'])

# Select features (X) and target variable (y)
X = country_df.index.year.values.reshape(-1, 1)
y = country_df['primary_energy_consumption']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the model's performance (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'The model predicts future energy consumption with an RMSE of {rmse:.2f} units.')


# Make predictions for future years
future_years = pd.DataFrame({'year': pd.date_range(start='2023', end='2030', freq='Y')})
future_years['year'] = future_years['year'].dt.year  # Extract year from datetime
future_predictions = model.predict(future_years[['year']].values.reshape(-1, 1))

# Plot the historical data and predictions
plt.figure(figsize=(12, 6))
plt.scatter(country_df.index, country_df['primary_energy_consumption'], label='Historical Data', color='blue')
plt.plot(X_test, y_pred, label='Test Set Predictions', color='green')
plt.plot(future_years['year'], future_predictions, label='Future Predictions', color='red', linestyle='--')
plt.title(f'Energy Consumption Prediction for {user_country}')
plt.xlabel('Year')
plt.ylabel('Primary Energy Consumption')
plt.legend()
plt.grid(True)
plt.show()

'''RMSE: Root Mean Squared Error'''

# Aggregating data for the whole world
world_df = df.groupby('year').sum()

# Select features (X) and target variable (y)
X_world = world_df.index.year.values.reshape(-1, 1)
y_world = world_df['primary_energy_consumption']

# Split the data into training and testing sets (not necessary for the whole world)
X_train_world, X_test_world, y_train_world, y_test_world = train_test_split(X_world, y_world, test_size=0.2, random_state=42)

# Create and train a linear regression model for the whole world
model_world = LinearRegression()
model_world.fit(pd.DataFrame(X_train_world, columns=['year']), y_train_world)

# Make predictions for the whole world
y_pred_world = model_world.predict(pd.DataFrame(X_test_world, columns=['year']))

# Calculate the model's performance for the whole world (Root Mean Squared Error)
rmse_world = np.sqrt(mean_squared_error(y_test_world, y_pred_world))
print(f'The world model predicts future energy consumption with an RMSE of {rmse_world:.2f} units.')


# Make predictions for future years for the whole world
future_years_world = pd.DataFrame({'year': range(2023, 2031)})
future_predictions_world = model_world.predict(future_years_world[['year']])


# Plot the historical data and predictions for the whole world
plt.figure(figsize=(12, 6))
plt.scatter(world_df.index, world_df['primary_energy_consumption'], label='Historical Data', color='blue')
plt.plot(X_test_world, y_pred_world, label='Test Set Predictions', color='green')
plt.plot(future_years_world['year'], future_predictions_world, label='Future Predictions', color='red', linestyle='--')
plt.title('World Energy Consumption Prediction')
plt.xlabel('Year')
plt.ylabel('Primary Energy Consumption')
plt.legend()
plt.grid(True)
plt.show()