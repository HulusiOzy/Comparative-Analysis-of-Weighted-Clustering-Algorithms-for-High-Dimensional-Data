import pandas as pd
import numpy as np
import sys
import os

def is_excel_file(filename):
    excel_extensions = ['.xlsx', '.xls', '.csv']
    return any(filename.lower().endswith(ext) for ext in excel_extensions) #Wrote this at 2AM, I also dont know what it does.

def excel_column_to_number(column_letter):
    ##For personal use
    ##Change if needed
    ##Convert to excel style column letters to number based
    ##A = 1, B = 2
    result = 0
    for char in column_letter.upper(): #Upper case for consistency
        result = result * 26 + (ord(char) - ord('A') + 1) #Example just incase I get confused in the future FOR B: result = 0 * 26 + (66 - 65 + 1) = 2
    return result - 1 #Treat excel colunms like a base-26 number system, go from 1 to 26 instead of 0 to 25

def number_to_excel_column(n): #Inverse excel_column_to_number()
    result = ""
    n += 1 #Convert to 1 based
    while n > 0:
        n, remainder = divmod(n - 1, 26) #Example: n = 54 should give 'BC' because divmod(54, 26) gives (2, 2), second it divmod(1,26) gives 0 1
        result = chr(65 + remainder) + result #chr(65 + 2) = 'C', chr(65 + 1) gives 'B'. So C -> result is C then B-> result is BC
    return result

def load_dataframe(input_file):
    file_extension = os.path.splitext(input_file)[1].lower() #just get the extension
    
    if file_extension == '.csv':
        return pd.read_csv(input_file)
    elif file_extension in ['.xlsx', '.xls']:
        return pd.read_excel(input_file)
    else:
        try: #I call this a hailmarry
            df = pd.read_csv(input_file, header=None) #Try reading CSV with numeric column names
            df.columns = range(df.shape[1]) #Number all columns from 0 to n-1
            return df
        except:
            print(f"Unable to read file '{input_file}'")
            sys.exit(1)

def get_target_column_index(df, target_input, is_excel=True):
    if not target_input:
        return len(df.columns) - 1
    
    if is_excel:
        return excel_column_to_number(target_input)
    try:
        return int(target_input) #For non excel files
    except ValueError:
        print(f"Invalid column number '{target_input}'")
        sys.exit(1)

#Y_{iv} = (X_{iv} - av) / bv
def is_categorical(column):
    return column.dtype == 'object' or column.dtype.name == 'category' #NOTE: 'category' is for categorical & 'object' is for mixed data

def process_categorical_column(column):
    dummy_cols = pd.get_dummies(column, prefix=column.name) #Saved by panda again
    for col in dummy_cols.columns: #Mean centering
        p = dummy_cols[col].mean()
        dummy_cols[col] = (dummy_cols[col] - p) / 1
    return dummy_cols

def process_numerical_column(column):
    av = column.mean()
    bv = column.max() - column.min() #No using standard deviation said the book
    return (column - av) / bv

def encode_target_column(column):
    if is_categorical(column):
        categories = column.unique() #I should find a way to use set here instead of .unique
        category_mapping = {cat: i for i, cat in enumerate(sorted(categories))} #Create map of unqiue cat's to numbers
        
        numerical_column = column.map(category_mapping) #Convert categories to numbers
        
        print("\nCategory to number mapping:") #Print the mapping for reference
        for cat, num in category_mapping.items():
            print(f"{cat} -> {num}")
            
        return numerical_column
    return column

input_file = input("Enter input file: (default: heart_failure_clinical_records_dataset.csv): ").strip() #.strip() :)
if not input_file:
    input_file = 'heart_failure_clinical_records_dataset.csv'

try: #This is what free time does to a person
    df = load_dataframe(input_file)
except FileNotFoundError:
    print(f"Error: Could not find file '{input_file}'")
    sys.exit(1)

column_style = input("Use letters or numbers for column selection? (L/N, default: L): ").strip().upper()
if not column_style:
    column_style = 'L'

target_input = input("Enter target column (default: last column): ").strip() #.strip() :)
if not target_input:
    target_column_index = len(df.columns) - 1
    target_column_letter = number_to_excel_column(target_column_index) if column_style == 'L' else str(target_column_index)
else:
    if column_style == 'L':
        target_column_index = excel_column_to_number(target_input)
    else:
        try:
            target_column_index = int(target_input)
        except ValueError:
            print(f"Error: Invalid column number '{target_input}'")
            sys.exit(1)

#First extract and write
target_column_name = df.columns[target_column_index]
target_series = df[target_column_name]
encoded_target = encode_target_column(target_series)
target_df = pd.DataFrame({target_column_name: encoded_target})
output_name = os.path.splitext(input_file)[0] + '.actual'
target_df.to_csv(output_name, index=False, header=False)
print(f"Target column '{target_column_name}' saved to {output_name}") #For debug :D

#Then remove
df_without_target = df.drop(columns=[target_column_name])

#AND FINALLY PROCESS
processed_columns = []
column_info = [] #New line to track column info
for column in df_without_target.columns:
    if is_categorical(df_without_target[column]):
        dummy_cols = process_categorical_column(df_without_target[column])
        processed_columns.append(dummy_cols) #No need to convert because already a df from .get_dummies
    else:
        processed_col = process_numerical_column(df_without_target[column])
        processed_columns.append(pd.DataFrame(processed_col, columns=[column])) #.concat needs all inputs as df's

#Changed it a little
df_processed = pd.concat(processed_columns, axis=1)
processed_output_name = input_file + '.data' #Keeping it simple
df_processed.to_csv(processed_output_name, index=False, header=False)










