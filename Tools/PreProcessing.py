import pandas as pd
import numpy as np
import sys
import os

class Preprocessing:
    def __init__(self, input_file=None, target_column=None, column_style='L'):

        self.input_file = input_file or 'heart_failure_clinical_records_dataset.csv'
        self.column_style = column_style.upper()
        self.target_column = target_column
        self.dataframe = None
        self.target_series = None
        self.processed_df = None
        
    def _is_excel_file(self, filename):
            ##For personal use
        ##Change if needed
        ##Convert to excel style column letters to number based
        ##A = 1, B = 2
        excel_extensions = ['.xlsx', '.xls', '.csv']
        return any(filename.lower().endswith(ext) for ext in excel_extensions)#Wrote this at 2AM, I also dont know what it does.
    
    def _excel_column_to_number(self, column_letter):
        ##For personal use
        ##Change if needed
        ##Convert to excel style column letters to number based
        ##A = 1, B = 2
        result = 0
        for char in column_letter.upper(): #Upper case for consistency
            result = result * 26 + (ord(char) - ord('A') + 1) #Example just incase I get confused in the future FOR B: result = 0 * 26 + (66 - 65 + 1) = 2
        return result - 1 #Treat excel colunms like a base-26 number system, go from 1 to 26 instead of 0 to 25
    
    def _number_to_excel_column(self, n): #Inverse excel_column_to_number()
        result = ""
        n += 1 #Convert to 1 based
        while n > 0:
            n, remainder = divmod(n - 1, 26) #Example: n = 54 should give 'BC' because divmod(54, 26) gives (2, 2), second it divmod(1,26) gives 0 1
            result = chr(65 + remainder) + result  #chr(65 + 2) = 'C', chr(65 + 1) gives 'B'. So C -> result is C then B-> result is BC
        return result
    
    def _load_dataframe(self):
        file_extension = os.path.splitext(self.input_file)[1].lower() #just get the extension
        
        try: #I call this a hailmarry
            if file_extension == '.csv':
                self.dataframe = pd.read_csv(self.input_file)
            elif file_extension in ['.xlsx', '.xls']:
                self.dataframe = pd.read_excel(self.input_file)
            else:
                self.dataframe = pd.read_csv(self.input_file, header=None) #Try reading CSV with numeric column names
                self.dataframe.columns = range(self.dataframe.shape[1])  #Number all columns from 0 to n-1
        #Added some proper error handling unlike the original code, feels like I should of done this project in java but eh
        except FileNotFoundError: 
            raise FileNotFoundError(f"Could not find file '{self.input_file}'")
        except Exception as e:
            raise ValueError(f"Unable to read file '{self.input_file}': {str(e)}")
    
    def _get_target_column_index(self):
        if not self.target_column:
            return len(self.dataframe.columns) - 1
        
        if self.column_style == 'L':
            return self._excel_column_to_number(self.target_column)
        try:
            return int(self.target_column) #For non excel files
        except ValueError:
            raise ValueError(f"Invalid column number '{self.target_column}'")
    
    #Y_{iv} = (X_{iv} - av) / bv
    def _is_categorical(self, column):
        #NOTE: 'category' is for categorical & 'object' is for mixed data
        return column.dtype == 'object' or column.dtype.name == 'category'
    
    def _process_categorical_column(self, column):
        dummy_cols = pd.get_dummies(column, prefix=column.name) #Saved by panda again
        for col in dummy_cols.columns: #Mean centering
            p = dummy_cols[col].mean()
            dummy_cols[col] = (dummy_cols[col] - p) / 1
        return dummy_cols
    
    def _process_numerical_column(self, column):
        av = column.mean()
        bv = column.max() - column.min() #No using standard deviation said the book
        return (column - av) / bv
    
    def _encode_target_column(self, column):
        if self._is_categorical(column):
            categories = column.unique() #I should find a way to use set here instead of .unique
            category_mapping = {cat: i for i, cat in enumerate(sorted(categories))} #Create map of unqiue cat's to numbers

            numerical_column = column.map(category_mapping) #Convert categories to numbers
            
            print("\nCategory to number mapping:") #Print the mapping for reference
            for cat, num in category_mapping.items():
                print(f"{cat} -> {num}")
                
            return numerical_column
        return column
    
    #Abit more unreadableness compared to original but eeeh
    def fit(self):
        self._load_dataframe()

        #First extract and write
        target_column_index = self._get_target_column_index()
        target_column_name = self.dataframe.columns[target_column_index]
        self.target_series = self.dataframe[target_column_name]
        encoded_target = self._encode_target_column(self.target_series)
        target_df = pd.DataFrame({target_column_name: encoded_target})
        target_output = os.path.splitext(self.input_file)[0] + '.actual'
        target_df.to_csv(target_output, index=False, header=False)

        #Then remove
        df_without_target = self.dataframe.drop(columns=[target_column_name])
        
        #AND FINALLY PROCESS
        processed_columns = []
        for column in df_without_target.columns:
            if self._is_categorical(df_without_target[column]):
                dummy_cols = self._process_categorical_column(df_without_target[column])
                processed_columns.append(dummy_cols) #No need to convert because already a df from .get_dummies
            else:
                processed_col = self._process_numerical_column(df_without_target[column])
                processed_columns.append(pd.DataFrame(processed_col, columns=[column])) #.concat needs all inputs as df's
        
        #Changed it a little
        self.processed_df = pd.concat(processed_columns, axis=1)
        processed_output = self.input_file + '.data' #Keeping it simple
        self.processed_df.to_csv(processed_output, index=False, header=False)
        
        return processed_output, target_output
    
    def save_predictions(self, output_filename=None):
        if self.processed_df is None:
            raise ValueError("Must call fit() before saving")
        
        if output_filename is None:
            output_filename = self.input_file + '.data'
            
        self.processed_df.to_csv(output_filename, index=False, header=False)

if __name__ == "__main__":
    preprocessor = Preprocessing()
    
    preprocessor.input_file = input("Enter input file (default: heart_failure_clinical_records_dataset.csv): ").strip() or 'heart_failure_clinical_records_dataset.csv'
    preprocessor.column_style = input("Use letters or numbers for column selection? (L/N, default: L): ").strip().upper() or 'L'
    preprocessor.target_column = input("Enter target column (default: last column): ").strip()
    
    processed_file, target_file = preprocessor.fit()
    print(f"\nProcessed data saved to: {processed_file}")
    print(f"Target values saved to: {target_file}")