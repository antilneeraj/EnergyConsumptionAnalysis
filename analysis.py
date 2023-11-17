import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

def perform_analysis(df, user_country):
    # Filter the DataFrame for the user-selected country
    country_df = df[df['country'] == user_country]

    # Set up subplots
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 12))

    # Energy Mix Trends
    energy_sources = ['coal', 'gas', 'renewables', 'nuclear']
    axes[0, 0].set_title(f'Energy Mix Trends for {user_country} Over the Years')
    for source in energy_sources:
        axes[0, 0].plot(country_df.index, country_df[f'{source}_share_energy'], label=source.capitalize())
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Share of Energy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Per Capita Consumption
    axes[0, 1].set_title(f'Per Capita Energy Consumption Trends for {user_country}')
    axes[0, 1].plot(country_df.index, country_df['per_capita_electricity'], marker='o', label='Per Capita Consumption')
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('Per Capita Electricity Consumption')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Renewable Energy Growth
    renewable_sources = ['solar', 'wind', 'hydro']
    axes[0, 2].set_title(f'Renewable Energy Growth for {user_country} Over the Years')
    for source in renewable_sources:
        axes[0, 2].plot(country_df.index, country_df[f'{source}_share_energy'], label=source.capitalize())
    axes[0, 2].set_xlabel('Year')
    axes[0, 2].set_ylabel('Share of Energy')
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    # Carbon Intensity
    axes[1, 0].set_title(f'Carbon Intensity of Electricity Generation for {user_country} Over the Years')
    axes[1, 0].plot(country_df.index, country_df['carbon_intensity_elec'], marker='o', color='red', label='Carbon Intensity')
    axes[1, 0].set_xlabel('Year')
    axes[1, 0].set_ylabel('Carbon Intensity')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Energy and GDP Relationship
    axes[1, 1].set_title(f'Energy Consumption vs. GDP for {user_country}')
    axes[1, 1].scatter(country_df['gdp'], country_df['primary_energy_consumption'], alpha=0.7)
    axes[1, 1].set_xlabel('GDP')
    axes[1, 1].set_ylabel('Primary Energy Consumption')
    axes[1, 1].grid(True)

    # Fossil Fuel Consumption
    fossil_fuels = ['coal', 'oil', 'gas']
    axes[1, 2].set_title(f'Fossil Fuel Consumption Trends for {user_country} Over the Years')
    for fuel in fossil_fuels:
        axes[1, 2].plot(country_df.index, country_df[f'{fuel}_consumption'], label=fuel.capitalize())
    axes[1, 2].set_xlabel('Year')
    axes[1, 2].set_ylabel('Fossil Fuel Consumption')
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    # Electricity Generation and Consumption
    axes[2, 0].set_title(f'Electricity Generation and Consumption Trends for {user_country} Over the Years')
    axes[2, 0].plot(country_df.index, country_df['electricity_generation'], label='Electricity Generation', marker='o')
    axes[2, 0].plot(country_df.index, country_df['energy_cons_change_twh'], label='Energy Consumption', marker='o')
    axes[2, 0].set_xlabel('Year')
    axes[2, 0].set_ylabel('TWH (Terawatt-hours)')
    axes[2, 0].legend()
    axes[2, 0].grid(True)

    # Future Energy Consumption Prediction
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
    axes[2, 1].set_title(f'Energy Consumption Prediction for {user_country}')
    axes[2, 1].scatter(country_df.index, country_df['primary_energy_consumption'], label='Historical Data', color='blue')
    axes[2, 1].plot(X_test, y_pred, label='Test Set Predictions', color='green')
    axes[2, 1].plot(future_years['year'], future_predictions, label='Future Predictions', color='red', linestyle='--')
    axes[2, 1].set_xlabel('Year')
    axes[2, 1].set_ylabel('Primary Energy Consumption')
    axes[2, 1].legend()
    axes[2, 1].grid(True)

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
    axes[2, 2].set_title('World Energy Consumption Prediction')
    axes[2, 2].scatter(world_df.index, world_df['primary_energy_consumption'], label='Historical Data', color='blue')
    axes[2, 2].plot(X_test_world, y_pred_world, label='Test Set Predictions', color='green')
    axes[2, 2].plot(future_years_world['year'], future_predictions_world, label='Future Predictions', color='red',
                    linestyle='--')
    axes[2, 2].set_xlabel('Year')
    axes[2, 2].set_ylabel('Primary Energy Consumption')
    axes[2, 2].legend()
    axes[2, 2].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def linear_regression_model(X_train, y_train):
    # Create and train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def calculate_rmse(y_true, y_pred):
    # Calculate the Root Mean Squared Error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse
