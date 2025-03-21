import pandas as pd
import numpy as np
import sys
import os

class Preprocessing:
    def __init__(self, input_file=None, target_column=None, column_style='L', bin_option=0, has_target=True):
        self.input_file = input_file or 'heart_failure_clinical_records_dataset.csv'
        self.column_style = column_style.upper()
        self.target_column = target_column
        self.bin_option = bin_option
        self.has_target = has_target
        self.dataframe = None
        self.target_series = None
        self.processed_df = None
        
    def _handle_csv_columns(self, operation='detect', column_letter=None, column_number=None):
        if operation == 'detect':
            csv_extensions = ['.csv', '.data', '.txt']
            return any(self.input_file.lower().endswith(ext) for ext in csv_extensions)
        elif operation == 'to_number':
            result = 0
            for char in column_letter.upper():
                result = result * 26 + (ord(char) - ord('A') + 1)
            return result - 1
        elif operation == 'to_letter':
            result = ""
            column_number += 1
            while column_number > 0:
                column_number, remainder = divmod(column_number - 1, 26)
                result = chr(65 + remainder) + result
            return result
    
    def _bin_target_column(self, column):
        n_unique = len(column.unique())
        
        if n_unique <= 5 or self.bin_option == 0:
            return column
            
        should_bin = False
        
        if self.bin_option == 1:
            should_bin = True
            bin_method = 1
            n_bins = 5
            
        elif self.bin_option == 2:
            if input(f"\nTarget column has {n_unique} unique values. Apply binning? (y/n): ").lower() == 'y':
                should_bin = True
                
                print("\nBinning methods:")
                print("1. Equal-width binning")
                print("2. Equal-frequency binning")
                print("3. User-specified binning")  # New option
                print("4. K-means binning")  # Moved from 3 to 4
                print("5. Quantile-based binning")  # Moved from 4 to 5
                
                try:
                    bin_method = int(input("\nSelect binning method (1-5, default: 1): ") or "1")
                    if bin_method < 1 or bin_method > 5:
                        bin_method = 1
                except ValueError:
                    bin_method = 1
                
                try:
                    n_bins_input = input(f"\nNumber of bins (default: 5): ").strip()
                    n_bins = int(n_bins_input) if n_bins_input else 5
                    if n_bins < 2:
                        n_bins = 5
                except ValueError:
                    n_bins = 5
        
        if not should_bin:
            return column
        
        numeric_column = pd.to_numeric(column, errors='coerce')
        
        if bin_method == 1:
            bins = np.linspace(numeric_column.min(), numeric_column.max(), n_bins + 1)
            labels = list(range(n_bins))  # Changed to use 0, 1, 2, etc.
            binned = pd.cut(numeric_column, bins=bins, labels=labels, include_lowest=True)
            print(f"\nEqual-width bins created: {[round(b, 2) for b in bins]}")
            
        elif bin_method == 2:
            bins = pd.qcut(numeric_column, q=n_bins, duplicates='drop')
            unique_bins = bins.cat.categories
            map_dict = {cat: i for i, cat in enumerate(unique_bins)}
            binned = bins.map(map_dict)
            print(f"\nEqual-frequency binning created {len(unique_bins)} bins")
            
        elif bin_method == 3:  # New user-specified binning
            try:
                print(f"\nSpecify {n_bins+1} cutoff points (including min and max values):")
                cutoffs = []
                for i in range(n_bins+1):
                    value = float(input(f"Cutoff point {i+1}: "))
                    cutoffs.append(value)
                
                # Sort cutoffs in case they weren't entered in order
                cutoffs.sort()
                
                # Create bins with the user-specified cutoffs
                labels = list(range(n_bins))
                binned = pd.cut(numeric_column, bins=cutoffs, labels=labels, include_lowest=True)
                print(f"\nUser-specified bins created: {[round(b, 2) for b in cutoffs]}")
            except ValueError as e:
                print(f"Error in user input: {e}. Using equal-width binning instead.")
                bins = np.linspace(numeric_column.min(), numeric_column.max(), n_bins + 1)
                labels = list(range(n_bins))
                binned = pd.cut(numeric_column, bins=bins, labels=labels, include_lowest=True)
                print(f"\nEqual-width bins created: {[round(b, 2) for b in bins]}")
        
        elif bin_method == 4:  # K-means binning (moved from option 3)
            try:
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=n_bins, random_state=0).fit(numeric_column.values.reshape(-1, 1))
                centers = sorted(kmeans.cluster_centers_.flatten())
                binned = pd.Series(kmeans.labels_, index=column.index)
                print(f"\nK-means binning with centers at approximately: {[round(c, 2) for c in centers]}")
            except ImportError:
                bins = np.linspace(numeric_column.min(), numeric_column.max(), n_bins + 1)
                labels = list(range(n_bins))
                binned = pd.cut(numeric_column, bins=bins, labels=labels, include_lowest=True)
                print(f"\nEqual-width bins created: {[round(b, 2) for b in bins]}")
        
        else:  # Quantile-based binning (moved from option 4)
            quantiles = [0]
            quantiles.extend([i/n_bins for i in range(1, n_bins)])
            quantiles.append(1)
            bins = numeric_column.quantile(quantiles).tolist()
            labels = list(range(n_bins))
            binned = pd.cut(numeric_column, bins=bins, labels=labels, include_lowest=True)
            print(f"\nQuantile-based bins at: {[round(q, 2) for q in bins]}")
        
        return binned
    
    def _is_categorical(self, column):
        return column.dtype == 'object' or column.dtype.name == 'category'
    
    def _process_column(self, column):
        if self._is_categorical(column):
            dummy_cols = pd.get_dummies(column, prefix=column.name)
            for col in dummy_cols.columns:
                p = dummy_cols[col].mean()
                dummy_cols[col] = (dummy_cols[col] - p) / 1
            return dummy_cols
        else:
            av = column.mean()
            bv = column.max() - column.min()
            if bv == 0:
                bv = 1
            processed_col = (column - av) / bv
            return pd.DataFrame(processed_col, columns=[column.name])
        
    def _load_and_process_data(self):
        try:
            file_extension = os.path.splitext(self.input_file)[1].lower()
            
            if file_extension == '.csv':
                self.dataframe = pd.read_csv(self.input_file)
            elif file_extension in ['.xlsx', '.xls']:
                self.dataframe = pd.read_excel(self.input_file)
            else:
                self.dataframe = pd.read_csv(self.input_file, header=None)
                self.dataframe.columns = range(self.dataframe.shape[1])
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find file '{self.input_file}'")
        except Exception as e:
            raise ValueError(f"Unable to read file '{self.input_file}': {str(e)}")
        
        target_output = None
        df_to_process = self.dataframe
        
        if self.has_target:
            if not self.target_column:
                target_column_index = len(self.dataframe.columns) - 1
            elif self.column_style == 'L':
                target_column_index = self._handle_csv_columns(operation='to_number', column_letter=self.target_column)
            else:
                try:
                    target_column_index = int(self.target_column)
                except ValueError:
                    raise ValueError(f"Invalid column number '{self.target_column}'")
                    
            target_column_name = self.dataframe.columns[target_column_index]
            self.target_series = self.dataframe[target_column_name]
            
            self.target_series = self._bin_target_column(self.target_series)
            
            categories = self.target_series.unique() if self._is_categorical(self.target_series) else None
            if categories is not None:
                category_mapping = {cat: i for i, cat in enumerate(sorted(categories))}
                encoded_target = self.target_series.map(category_mapping).fillna(0)
                print("\nCategory to number mapping:")
                for cat, num in category_mapping.items():
                    print(f"{cat} -> {num}")
            else:
                encoded_target = self.target_series
                
            target_df = pd.DataFrame({target_column_name: encoded_target})
            target_output = os.path.splitext(self.input_file)[0] + '.actual'
            target_df.to_csv(target_output, index=False, header=False)
            
            df_to_process = self.dataframe.drop(columns=[target_column_name])
        
        processed_columns = []
        
        for column_name in df_to_process.columns:
            column = df_to_process[column_name]
            processed_columns.append(self._process_column(column))
        
        self.processed_df = pd.concat(processed_columns, axis=1)
        processed_output = self.input_file + '.data'
        self.processed_df.to_csv(processed_output, index=False, header=False)
        
        return processed_output, target_output
    
    def fit(self):
        return self._load_and_process_data()
    
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
    
    bin_choice = input("Binning option (0=none, 1=auto equal-width, 2=prompt for method, default: 0): ").strip()
    preprocessor.bin_option = int(bin_choice) if bin_choice else 0
    
    has_target = input("Does the file have a target column? (y/n, default: y): ").lower().strip() != 'n'
    preprocessor.has_target = has_target
    
    processed_file, target_file = preprocessor.fit()
    print(f"\nProcessed data saved to: {processed_file}")
    if target_file:
        print(f"Target values saved to: {target_file}")