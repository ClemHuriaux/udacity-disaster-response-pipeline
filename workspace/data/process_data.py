import sys
import pandas as pd
import sqlalchemy as db


def load_data(messages_filepath, categories_filepath):
    """ Load the data from the csv files into dataframes

    INPUTS:
    -------
    messages_filepath: str
        The path of the csv files with the messages
    categories_filepath: str
        The path of the csv files with the categories

    RETURNS:
    --------
    DataFrame: A merge dataframe of the two dataframe we built
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages.merge(categories, how="left")


def clean_data(df):
    """ Clean the data to use it
    We do several cleaning here in order to use the dataset later for the model

    INPUTS:
    -------
    df: DataFrame
        The DataFrame we want to clean

    RETURNS:
    --------
    df: DataFrame
        The clean DataFrame we gave in input
    """

    # Split categories into separate columns
    categories = df.categories.str.split(";", expand=True)
    row = categories.iloc[0]
    category_colnames = row.unique()
    categories.columns = category_colnames
    categories = categories.rename(columns=lambda x: x[:-2])

    # convert category values to 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1:]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # Replace catgories columun in df with the new categories columns
    df.drop("categories", axis=1, inplace=True)
    df = pd.concat([df, categories], sort=False, axis=1)

    # Remove Duplicates
    df.drop_duplicates(keep='first', inplace=True)

    # Handle the value 2 in the related column.
    df[df.related == 2] = 0  # Put it to 0 since the category is not valid in this case
    return df


def save_data(df, database_filename):
    """ Save the dataframe we gave into Database with the name we want

    INPUTS:
    -------
    df: DataFrame
        The DataFrame we want to save
    database_filename: str
        The name of the database we want
   """

    engine = db.create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster_response', engine, index=False, if_exists="replace")


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
