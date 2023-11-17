import pandas as pd
from analysis import perform_analysis

def main():
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

    # Perform analysis
    perform_analysis(df, user_country)

if __name__ == "__main__":
    main()