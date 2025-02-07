import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class ARI:
    def __init__(self):
        self.contingency_table = None
        self.score = None
        

    def _to_numpy_array(self, input_data):
        '''
        I cant believe it came down to this
        '''
        if isinstance(input_data, np.ndarray):
            return input_data
            
        if isinstance(input_data, (pd.Series, pd.DataFrame)):
            return input_data.to_numpy()
            
        if isinstance(input_data, (list, tuple)):
            return np.array(input_data)
            
        raise TypeError(f"Unsupported input type: {type(input_data)}. Must be numpy array, pandas Series/DataFrame, list, or tuple") #Adding this for future hulus :D

    def _create_contingency(self, U, V):
        #Mental Note: We need unique classes from both partitions to create a comparison matrix
        u_classes = np.unique(U)
        v_classes = np.unique(V)
        
        n_rows = len(u_classes)
        n_cols = len(v_classes)
        table = np.zeros((n_rows, n_cols), dtype=int)
        
        #Mental Note: Fill table by counting elements that appear in both classes
        for i, u_class in enumerate(u_classes):
            for j, v_class in enumerate(v_classes):
                table[i, j] = np.sum((U == u_class) & (V == v_class))
                
        return table

    def _calculate_pair_counts(self, contingency):
        #Mental Note: Calculate basic counts needed for multiple formulas
        n = np.sum(contingency)
        sum_squares = np.sum(contingency ** 2)
        
        #Mental Note: Calculate row and column sums for pair agreement calculation
        row_sums = np.sum(contingency, axis=1)
        col_sums = np.sum(contingency, axis=0)
        sum_squares_rows = np.sum(row_sums ** 2)
        sum_squares_cols = np.sum(col_sums ** 2)
        
        #Mental Note: Implement paper equations for pair counts
        a = (sum_squares - n) / 2.0
        b = (sum_squares_rows - sum_squares) / 2.0
        c = (sum_squares_cols - sum_squares) / 2.0
        d = (sum_squares + n**2 - sum_squares_rows - sum_squares_cols) / 2.0
        
        return a, b, c, d

    def _calculate_ari(self, contingency):
        #Mental Note: Get pair counts for different agreement scenarios
        a, b, c, d = self._calculate_pair_counts(contingency)
        
        #Mental Note: Calculate components for ARI formula from paper
        n_choose_2 = a + b + c + d
        numerator = (n_choose_2 * (a + d)) - ((a + b) * (a + c) + (c + d) * (b + d))
        denominator = (n_choose_2 * n_choose_2) - ((a + b) * (a + c) + (c + d) * (b + d))
        
        if denominator == 0:
            return 0.0
            
        return numerator / denominator

    def fit(self, actual, predicted):
        #Mental Note: Main method to calculate ARI score

        if not isinstance(actual, np.ndarray):
            actual = np.array(actual)
        if not isinstance(predicted, np.ndarray):
            predicted = np.array(predicted)

        self.contingency_table = self._create_contingency(actual, predicted)
        self.score = self._calculate_ari(self.contingency_table)
        return self.score

    def plot(self, title=None):
        #Mental Note: Visualization method for analysis
        if self.contingency_table is None:
            return
            
        plt.figure(figsize=(6, 4))
        plt.rcParams['figure.facecolor'] = 'white'
        
        im = plt.imshow(self.contingency_table, cmap='Blues')
        
        height, width = self.contingency_table.shape
        for i in range(height):
            for j in range(width):
                color = 'white' if self.contingency_table[i, j] > self.contingency_table.max()/2 else 'black'
                plt.text(j, i, str(self.contingency_table[i, j]),
                        ha='center', va='center', color=color)
        
        plt.xlabel('Predicted Classes')
        plt.ylabel('Actual Classes')
        if title:
            plt.title(title)
        else:
            plt.title(f"Contingency Table (ARI = {self.score:.4f})")
        
        plt.xticks(range(width))
        plt.yticks(range(height))
        
        cbar = plt.colorbar(im)
        cbar.set_label('Number of Samples')
        
        plt.show()

if __name__ == "__main__":
    actual_labels = pd.read_csv('output.actual', header=None).iloc[:, 0].values
    predicted_labels = pd.read_csv('output.predicted', header=None).iloc[:, 0].values
    
    ari_calculator = ARI()
    score = ari_calculator.fit(actual_labels, predicted_labels)
    print(f"ARI Score: {score}")
    ari_calculator.plot()
