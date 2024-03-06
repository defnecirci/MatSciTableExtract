import json
import Levenshtein as lev
import os
from collections import defaultdict
import re
import numpy as np
import scipy



def find_ground_truth_and_predicted_files(base_path, folder_name):
    matched_pairs = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if 'groundtruth' in root and file.endswith('.txt'):
                ground_truth_file = os.path.join(root, file)
                unique_identifier = file[:6]
                predicted_folder = os.path.join(root.replace('groundtruth', folder_name))
                if os.path.isdir(predicted_folder):
                    for predicted_file in os.listdir(predicted_folder):
                        if predicted_file.startswith(unique_identifier) and predicted_file.endswith('.txt'):
                            matched_predicted_file = os.path.join(predicted_folder, predicted_file)
                            matched_pairs.append((ground_truth_file, matched_predicted_file))
                            break
    return matched_pairs

def read_json_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            # Remove comments (anything after '//' until the end of line)
            content_no_comments = re.sub(r'//.*', '', content)
            return json.loads(content_no_comments)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        print(f"JSON decode error in {file_path}: {e}")
    except Exception as e:
        print(f"An error occurred while reading {file_path}: {e}")
    return None

def get_mean_and_ci_h(data, confidence=0.95):
    data_arr = 1.0 * np.array(data)
    data_n = len(data_arr)
    data_mean, data_se = np.mean(data_arr), scipy.stats.sem(data_arr)
    ci_h = data_se * scipy.stats.t.ppf((1 + confidence) / 2., data_n-1)
    return data_mean, ci_h

def calculate_sample_f1(data_props, truth_props, threshold=0.4):
    TP, FP, FN = 0, 0, 0
    matched = []
    unmatched_data = []
    ground_truth_list = [(name, details) for name, details in truth_props.items()]
    matched_truth_indices = set()

    for data_name, data_details in data_props.items():
        closest_match_index = None
        closest_distance = float('inf')

        data_str = f"{data_name.lower()}"
        

        for i, (truth_name, truth_details) in enumerate(ground_truth_list):

            if i in matched_truth_indices:
                continue

            truth_str = f"{truth_name.lower()}"
            


            distance_ratio = lev.distance(data_str, truth_str) / max(len(data_str), len(truth_str))
            print("lala",data_str,truth_str)
            print("distance_ratio",distance_ratio)

            if distance_ratio < closest_distance:
                closest_distance = distance_ratio
                closest_match_index = i
        #print("closest_distance",closest_distance)
        if closest_distance < (1 - threshold):
            matched_truth_name, _ = ground_truth_list[closest_match_index]
            matched.append((data_name, matched_truth_name))
            TP += 1
            matched_truth_indices.add(closest_match_index)
        else:
            unmatched_data.append(data_name)
            FP += 1

    FN = len(ground_truth_list) - len(matched_truth_indices)

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return f1_score, matched, unmatched_data, [ground_truth_list[i][0] for i in range(len(ground_truth_list)) if i not in matched_truth_indices], TP, FP, FN

def calculate_f1_scoreswithoutexcludingtables(data, ground_truth):
    f1_scores = []
    all_matched = []
    all_unmatched_data = []
    all_unmatched_truth = []
    all_TP = []
    all_FP = []
    all_FN = []

    # Iterate over each sample in ground truth
    for index, truth_item in enumerate(ground_truth):
        if index < len(data):
            data_item = data[index]
            if 'properties' in data_item and 'properties' in truth_item:
                sample_f1, matched, unmatched_data, unmatched_truth, TP, FP, FN = calculate_sample_f1(data_item['properties'], truth_item['properties'])
            else:
                # Handle missing 'properties'
                sample_f1, matched, unmatched_data, unmatched_truth, TP, FP, FN = 0, [], [], list(truth_item.get('properties', {}).keys()), 0, 0, len(truth_item.get('properties', {}))
        else:
            # Handle completely missing data sample
            sample_f1, matched, unmatched_data, unmatched_truth, TP, FP, FN = 0, [], [], list(truth_item.get('properties', {}).keys()), 0, 0, len(truth_item.get('properties', {}))

        f1_scores.append(sample_f1)
        all_matched.append(matched)
        all_unmatched_data.append(unmatched_data)
        all_unmatched_truth.append(unmatched_truth)
        all_TP.append(TP)
        all_FP.append(FP)
        all_FN.append(FN)

    average_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    return f1_scores, average_f1, all_matched, all_unmatched_data, all_unmatched_truth, all_TP, all_FP, all_FN


current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
data_path = os.path.join(parent_dir, 'data')
folder_names = ["imageoutput3", "StructuredFormatoutput", "OCRoutput", "StructuredFormatwithCaptionsoutput"]



overall_results = {}
for folder_name in folder_names:
    print(f"\nProcessing Folder: {folder_name}")
    matched_files = find_ground_truth_and_predicted_files(data_path, folder_name)
    total_f1_score = 0
    total_samples = 0
    Table_F1s=[]
    
    for ground_truth_file, predicted_file in matched_files:
        print("\nProcessing Ground Truth File:", ground_truth_file)
        data = read_json_from_file(predicted_file)
        ground_truth = read_json_from_file(ground_truth_file)
   

        if data is not None and ground_truth is not None:

            sample_f1_scores, average_f1, all_matched, all_unmatched_data, all_unmatched_truth, all_TP, all_FP, all_FN = calculate_f1_scoreswithoutexcludingtables(data, ground_truth)
            
            print("F1 Scores for each sample:", sample_f1_scores)
            print("Average F1 Score for this folder:", average_f1)
            Table_F1s.append(average_f1)


            total_f1_score += sum(sample_f1_scores)
            total_samples += len(sample_f1_scores)

            for i in range(len(sample_f1_scores)):
                print(f"\nSample {i+1}:")
                print("True Positives (TP):", all_TP[i])
                print("False Positives (FP):", all_FP[i])
                print("False Negatives (FN):", all_FN[i])
                if i < len(all_matched):
                    print("Matched Properties:", all_matched[i])
                if i < len(all_unmatched_data):
                    print("Unmatched Data Properties:", all_unmatched_data[i])
                if i < len(all_unmatched_truth):
                    print("Unmatched Ground Truth Properties:", all_unmatched_truth[i])
        else:
            print("Error in loading JSON data from", ground_truth_file, "and", predicted_file)

    print("Table_F1s",Table_F1s)
    print(np.mean([val for val in Table_F1s if val != None]))
    print("CI",get_mean_and_ci_h([val for val in Table_F1s if val != None])[1])


    overall_average_f1 = total_f1_score / total_samples if total_samples > 0 else 0
    overall_results[folder_name] = {
            "average_f1": overall_average_f1,
            "sample_count": total_samples
        }

