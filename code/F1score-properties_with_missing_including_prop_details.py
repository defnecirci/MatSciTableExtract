import json
import Levenshtein as lev
import os
from collections import defaultdict
import re
import matplotlib.pyplot as plt
import seaborn as sns
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


def calculate_detailed_score(data_details, truth_details):
    # Calculate a score based on the similarity of value, unit, and conditions
    # Return a score between 0 and 1, where 1 is a perfect match


    value_match = compare_values(data_details.get('value'), truth_details.get('value'))
    print("value_match",value_match)
    unit_match = compare_units(data_details.get('unit'), truth_details.get('unit'))
    print("unit_match",unit_match)
    conditions_match = compare_conditions(data_details.get('conditions', []), truth_details.get('conditions', []))
    print("conditions_match",conditions_match)
    return (value_match + unit_match + conditions_match) / 3




def compare_values(data_value, truth_value):
    # Exact match check
    return 1 if data_value == truth_value else 0

def compare_units(data_unit, truth_unit):
    # Exact match check
    return 1 if data_unit == truth_unit else 0

def compare_conditions(data_conditions, truth_conditions):
    if not data_conditions and not truth_conditions:
        return 1  # Both are empty or None

    if not data_conditions or not truth_conditions:
        return 0  # One is empty or None, the other is not

    total_score = 0
    matched_indices = set()

    # Iterate over each condition in data_conditions
    for data_condition in data_conditions:
        best_match_score = 0
        best_match_index = None

        # Compare with each condition in truth_conditions
        for i, truth_condition in enumerate(truth_conditions):
            if i in matched_indices:
                continue

            match_score = condition_match_score(data_condition, truth_condition)
            if match_score > best_match_score:
                best_match_score = match_score
                best_match_index = i

        if best_match_index is not None:
            matched_indices.add(best_match_index)

        total_score += best_match_score

    # Normalize the score based on the number of conditions
    max_conditions = max(len(data_conditions), len(truth_conditions))
    print("total_score",total_score,"max_conditions",max_conditions)
    normalized_score = total_score / max_conditions if max_conditions > 0 else 0
    print("conditions score",normalized_score )
    return normalized_score

def condition_match_score(data_condition, truth_condition):
    # Compare individual aspects of the conditions and calculate a score
    type_match = 1 if data_condition.get('type') == truth_condition.get('type') else 0
    value_match = 1 if data_condition.get('value') == truth_condition.get('value') else 0
    unit_match = 1 if data_condition.get('unit') == truth_condition.get('unit') else 0

    return (type_match + value_match + unit_match) / 3



def get_differences(data_name, data_details, truth_details):
    # Compare and return the differences in a readable format
    diff = f"Differences in '{data_name}':\n"
    diff += f"  Predicted Output: Value: {data_details.get('value')}, Unit: {data_details.get('unit')}, Conditions: {data_details.get('conditions')}\n"
    diff += f"  Ground Truth: Value: {truth_details.get('value')}, Unit: {truth_details.get('unit')}, Conditions: {truth_details.get('conditions')}"
    return diff

def calculate_sample_f1(data_props, truth_props, threshold=0.4, detailed_threshold=1):
    TP, FP, FN = 0, 0, 0
    
    matched = []
    unmatched_data = []
    differences = [] 
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

            if distance_ratio < closest_distance:
                closest_distance = distance_ratio
                closest_match_index = i


        if closest_distance < (1 - threshold):
            _, matched_truth_details = ground_truth_list[closest_match_index]
            detailed_score = calculate_detailed_score(data_details, matched_truth_details)
            matched_truth_name, _ = ground_truth_list[closest_match_index]

            if detailed_score >= detailed_threshold:
                matched.append((data_name, matched_truth_name))
                matched_truth_indices.add(closest_match_index)
                TP += 1
            else:
                FP += 1
                
                diff = get_differences(data_name, data_details, matched_truth_details)
                differences.append(diff)
        else:
            unmatched_data.append(data_name)
            FP += 1
            differences.append(f"Unmatched property: {data_name}")

    
    FN = len(ground_truth_list) - len(matched_truth_indices)
    print("Differences:", differences)


    precision = TP / (TP + FP) if TP + FP > 0 else 0

    print("precision",precision)
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    print("recall",recall)
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



def calculate_f1_scoreswithexcludingtables(data, ground_truth):
    f1_scores = []
    all_matched = []
    all_unmatched_data = []
    all_unmatched_truth = []
    all_TP = []
    all_FP = []
    all_FN = []
    any_comparable = False  # Flag to check if there's any comparable sample

    # First pass: Check for any comparable samples
    for data_item, truth_item in zip(data, ground_truth):
        if 'properties' in data_item and 'properties' in truth_item:
            any_comparable = True
            break

    # Only calculate F1 scores if there are comparable samples
    if any_comparable:
        for index, truth_item in enumerate(ground_truth):
            if index < len(data):
                data_item = data[index]
                if 'properties' in data_item and 'properties' in truth_item:
                    sample_f1, matched, unmatched_data, unmatched_truth, TP, FP, FN = calculate_sample_f1(data_item['properties'], truth_item['properties'])
                else:
                    # Assign an F1 score of 0 if 'properties' is missing in OCR output
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

        average_f1 = sum(f1_scores) / len(f1_scores)
    else:
        # No comparable samples, so no F1 scores to calculate
        average_f1 = None

    return f1_scores, average_f1, all_matched, all_unmatched_data, all_unmatched_truth, all_TP, all_FP, all_FN




current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
data_path = os.path.join(parent_dir, 'data')
folder_names = ["imageoutput3", "StructuredFormatoutput", "OCRoutput","StructuredFormatwithCaptionsoutput"]
overall_results = {}

# Iterate over each folder name
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


    print("Table_F1s",Table_F1s,"LEN",len(Table_F1s))
    print(np.mean([val for val in Table_F1s if val != None]))
    print("CI",get_mean_and_ci_h([val for val in Table_F1s if val != None])[1])

    overall_average_f1 = total_f1_score / total_samples if total_samples > 0 else 0
    overall_results[folder_name] = {
        "average_f1": overall_average_f1,
        "sample_count": total_samples
    }


