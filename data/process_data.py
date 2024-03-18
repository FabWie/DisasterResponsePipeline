import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    The function read messages & categories data and merges the data into a single dataframe.
    :param messages_filepath: path to messages data
    :param categories_filepath: path to categories data
    :return: dataframe
    """
    messages = pd.read_csv(messages_filepath) #read messages data
    categories = pd.read_csv(categories_filepath) #read categories data
    df = pd.merge(messages, categories, on="id") #merge messages & categories data
    return df

def clean_data(df):
    """
    The function creates first a dataframe with 36 individual categories and is later merged with the message data
    into a clean dataframe.
    :param df: name of the dataframe
    :return: cleaned dataframe
    """
    categories = df["categories"].str.split(";", expand=True) #create dataframe with 36 individual category columns
    row = categories.iloc[0] #select first row
    category_colnames = row.apply(lambda x: x[:-2]) #use row as new cols name
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str[-1] # set each value to be the last character of the string
        categories[column] = categories[column].astype(int) # convert column from string to numeric
    df.drop(columns="categories", inplace=True) #drop column categories
    df.drop_duplicates(inplace=True) # drop duplicates
    clean_df = pd.concat([df, categories], axis=1, join="inner")
    return clean_df

def save_data(df, database_filename):
    """
    The function creates a database and the related Table based on a dataframe.
    :param df: dataframe
    :param database_filename: name of the database
    :return:
    """
    engine = create_engine('sqlite:///' + str(database_filename))
    df.to_sql('Messages_Categories_Table', engine, if_exists='replace', index=False)

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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
    print("process_data.py executed successfully.")

    # cd data
    # python process_data.py messages.csv categories.csv DisasterResponse.db