
'''
Try to scrape the information from the website: 
'''
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from io import StringIO
import time
import random


def get_athlete_dict(soup, id):
    table = soup.find("table", attrs = {"class": "biodata"})
    df = pd.read_html(StringIO(str(table)), index_col = 0, dtype_backend = "pyarrow")[0]
    print("------- df ---------")
    print(df)
    output_df = df.T
    output_df['athlete_id'] = id
    print("-------- output df --------")
    print(output_df)
    return output_df

def get_athlete_results(soup, id):
    table = soup.find("table", {"class": "table"})
    df = pd.read_html(StringIO(str(table)))[0]
    print("------- df ---------")
    print(df)

    # Assigning new columns based on the row index
    df['athlete_id'] = id
    df['NOC'] = None
    df['Discipline'] = None

    rows_to_keep = df.index[df['Games'].isna()].tolist()
    rows_with_noc = df.index[~df['Games'].isna()].tolist()

    columns_to_ffill = ['Games', 'NOC', 'As', 'Discipline']
    df[columns_to_ffill] = df[columns_to_ffill].ffill()
    print("------ df (point 0) ---------")
    print(df)
    # Rename 
    df.rename(columns={'Discipline (Sport) / Event': 'Event', 'NOC / Team': 'Team'}, inplace = True)

    print("------ df (point 1) -------- ")
    print(df)
    df.drop(columns=['Unnamed: 6'], inplace=True, errors='ignore')

    print("------ df (point 2) --------")
    print(df)
    return df.iloc[rows_to_keep]


if __name__ == "__main__":
    base_url = "https://www.olympedia.org/athletes"

    SIZE = 200000
    columns = ['Roles', 'Sex', 'Full name', 'Used name', 'Born', 'Died', 'NOC', 'athlete_id']
    output = pd.DataFrame(columns = columns)
    results = pd.DataFrame()
    errors = []

    for i in range(1, SIZE):
        if (1 % 1000 == 0 and i != 0):
            print(i)
            results.to_csv(f'results/results_{i}.csv', index = False)
            output.to_csv(f'athletes/bios_{i}.csv', index = False)
        elif i % 250 == 0:
            print(i)
        try:
            # Send a GET request to the website
            athelete_url = f"{base_url}/{i}"
            response = requests.get(athelete_url, timeout= 60)
            
            if (response.status_code == 200):
                print(response)
                
                soup = BeautifulSoup(response.content, "html.parser")

                # TODO: scrapping logic
                df = get_athlete_dict(soup, i)
                output = pd.concat([output if not output.empty else None, df])

                result = get_athlete_results(soup, i)
                results = pd.concat([results if not results.empty else None, result])
            
            else:
                errors.append(i)
                print("Failed to retrieve the webpage. Status code:", response.status_code)
                print(f"index {i}")
        except Exception as e:
            errors.append(i)
            print(f"Error for index {i}")
            print(e)
    output.to_csv('bios.csv', index = False)
    results.to_csv('results.csv', index = False)

    with open("errors_list.txt", "w") as output:
        output.write(str(errors))
    
        

    
