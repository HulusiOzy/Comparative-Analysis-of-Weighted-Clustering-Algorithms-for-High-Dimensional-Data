import numpy as np
import pandas as pd
import random
import heapq
import os
from itertools import permutations
from collections import defaultdict

from Algorithms.Straight_K_Means import KMeans
from Algorithms.AnomalousPattern import AnomalousPattern
from Algorithms.I_KMeans import IKMeans
from Algorithms.W_KMeans import WKMeans
from Algorithms.SW_KMeans import SWKMeans
from Algorithms.UPGMA import UPGMA
from Algorithms.Wards_Method import Wards
from Algorithms.Wards_Method_Heap import WardsHeap
from Tools.PreProcessing import Preprocessing
from Tools.Silhouette_Index import SilhouetteIndex
from Tools.Table_of_Confusion import TableOfConfusion
from Metrics.MinkowskiMetric import MinkowskiMetric

#
# THIS JUST EXISTS FOR TESTING EASILY
#

algos = {1: KMeans, 2: AnomalousPattern, 3: IKMeans, 4: WKMeans, 
         5: SWKMeans, 6: UPGMA, 7: Wards, 8: WardsHeap}

def cleanup_generated_files(dataset_name):
    base_filename = dataset_name.split('.')[0]
    generated_extensions = [
        '.data',
        '.actual',
        '.predicted'
    ]    
    for ext in generated_extensions:
        filepath = f"{base_filename}{ext}"
        if os.path.exists(filepath):
                os.remove(filepath)

    #To clean up the temporary dataset copy if it exists
    if os.path.exists(dataset_name) and dataset_name.startswith(tuple(['iris', 'search', 'performance', 'depression', 'heart'])):
        print(f"Removed temporary dataset copy: {dataset_name}")

def write_cluster_assignments(labels, output_filename):
    with open(output_filename, 'w') as f:
        for label in labels:
            f.write(f"{label}\n")

def load_datasets():
    datasets = {'iris': 'Datasets/iris.data', 'search': 'Datasets/search_engine_data.csv',
               'performance': 'Datasets/StudentPerformanceFactors.csv',
               'depression': 'Datasets/Depression Student Dataset.csv',
               'heart': 'Datasets/heart_failure_clinical_records_dataset.csv'}
    dfs = {}
    for name, path in datasets.items():
        dfs[name] = pd.read_csv(path)
    return dfs

def pre_processing():
    df_names = {'iris': 'iris.data', 'search': 'search_engine_data.csv',
                'performance': 'StudentPerformanceFactors.csv',
                'depression': 'Depression Student Dataset.csv',
                'heart': 'heart_failure_clinical_records_dataset.csv'}
    print("1. iris.data")
    print("2. search_engine_data.csv")
    print("3. StudentPerformanceFactors.csv")
    print("4. Depression Student Dataset.csv")
    print("5. heart_failure_clinical_records_dataset.csv")
    dataset_choice = int(input())
    selected_dataset = list(df_names.values())[dataset_choice - 1]
    dfs = load_datasets()
    dfs[list(df_names.keys())[dataset_choice - 1]].to_csv(selected_dataset, index=False, header=False)
    preprocessor = Preprocessing(selected_dataset)
    processed_file, actual_file = preprocessor.fit()
    actual_labels = pd.read_csv(actual_file, header=None)
    number_of_clusters = len(set(actual_labels[0]))
    return processed_file, number_of_clusters, selected_dataset

def run_algos(dataset_filename, selected_dataset, number_of_clusters):
    print("1. KMeans")
    print("2. AnomalousPattern")
    print("3. IKMeans")
    print("4. WKMeans")
    print("5. SWKMeans")
    print("6. UPGMA")
    print("7. Wards")
    print("8. WardsHeap")
    algorithm_choice = int(input())
    selected_algorithm = algos[algorithm_choice]
    
    if algorithm_choice in [4, 5]:#S/WKMeans
        print("Beta value:")
        beta_value = float(input())
        algorithm_instance = selected_algorithm(filename=dataset_filename, k=number_of_clusters, beta=beta_value)
    elif algorithm_choice in [1, 3]:  #Means, IKMeans
        algorithm_instance = selected_algorithm(filename=dataset_filename, k=number_of_clusters)
    elif algorithm_choice == 2:  #AnomalousPattern
        algorithm_instance = selected_algorithm(filename=dataset_filename)
    else:  #UPGMA, Wards, WardsHeap
        algorithm_instance = selected_algorithm(filename=dataset_filename, n_clusters=number_of_clusters)
        
    cluster_labels = algorithm_instance.fit()
    base_filename = selected_dataset.split('.')[0]
    output_filename = f"{base_filename}.predicted"
    write_cluster_assignments(cluster_labels, output_filename)
    return cluster_labels

def run_stats(selected_dataset):
    base_filename = selected_dataset.split('.')[0]
    data_file = f"{selected_dataset}.data"
    predicted_file = f"{base_filename}.predicted"
    actual_file = f"{base_filename}.actual"
    confusion = TableOfConfusion(actual_file, predicted_file)
    accuracy = confusion.fit()
    silhouette = SilhouetteIndex(data_file, predicted_file)
    silhouette_score = silhouette.fit()
    print(f"Silhouette: {silhouette_score}")
    return accuracy, silhouette_score

if __name__ == "__main__":
    try:
        processed_dataset, number_of_clusters, selected_dataset = pre_processing()
        cluster_labels = run_algos(processed_dataset, selected_dataset, number_of_clusters)
        accuracy, silhouette = run_stats(selected_dataset)
    finally:
        # Clean up generated files even if an error occurs
        cleanup_generated_files(selected_dataset)