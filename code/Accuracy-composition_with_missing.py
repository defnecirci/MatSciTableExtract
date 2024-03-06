import json
import os
import re
import numpy as np
import scipy

# Function to read JSON data from a file
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

# Compute confidence interval, middle, lower, and upper
def get_mean_and_ci_h(data, confidence=0.95):
    data_arr = 1.0 * np.array(data)
    data_n = len(data_arr)
    data_mean, data_se = np.mean(data_arr), scipy.stats.sem(data_arr)
    ci_h = data_se * scipy.stats.t.ppf((1 + confidence) / 2., data_n-1)
    return data_mean, ci_h

def compare_ignore_percent(value1, value2):
    # Remove '%' and strip whitespace
    str_value1 = str(value1).replace('%', '').strip()
    str_value2 = str(value2).replace('%', '').strip()

    # Try converting to float for numeric comparison, else default to string comparison
    try:
        num_value1 = float(str_value1)
        num_value2 = float(str_value2)
        return num_value1 == num_value2
    except ValueError:
        # Fallback to case-insensitive string comparison if conversion to float fails
        return str_value1.lower() == str_value2.lower()


def compare_composition(comp1, comp2):
    # Function to safely convert value to float after ensuring it's a string
    def safe_float_conversion(value):
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            return float(value.replace('%', '').strip())
        else:
            raise ValueError("Invalid value for conversion")

    # Initialize variables to handle potential conversion failures
    amount1, amount2 = None, None

    # Try to convert amounts to float
    try:
        amount1 = safe_float_conversion(comp1.get("amount", "0"))
    except ValueError:
        pass  # If conversion fails, amount1 remains None

    try:
        amount2 = safe_float_conversion(comp2.get("amount", "0"))
    except ValueError:
        pass  # If conversion fails, amount2 remains None

    # Check if both amounts are zero
    if amount1 is not None and amount2 is not None:
        if amount1 == 0 and amount2 == 0:
            return 1  # If both amounts are zero, return full accuracy

    # If conversion to float failed or amounts are not both zero, continue with normal comparison
    correct_fields = 0
    total_fields = 2  # amount and type

    # Check amount
    if compare_ignore_percent(comp1.get("amount", ""), comp2.get("amount", "")):
        correct_fields += 1

    # Check type
    if compare_ignore_percent(comp1.get("type", ""), comp2.get("type", "")):
        correct_fields += 1

    return correct_fields / total_fields



def compare_dicts_and_print(dict1, dict2, fields_to_check):
    # Validate input types
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        print("One of the inputs is not a dictionary. Skipping comparison.")
        print("dict1:", dict1)
        print("dict2:", dict2)
        return None

 

    differences = {}

    for key in fields_to_check:
        if key in dict1 or key in dict2:
            if key == "composition":
                # Special handling for "composition" field
                composition_accuracy = compare_composition(dict1.get(key, {}), dict2.get(key, {}))
                if composition_accuracy < 1:
                    differences[key] = {
                        "first": dict1.get(key, {}),
                        "second": dict2.get(key, {}),
                        "accuracy": composition_accuracy
                    }
            else:
                if key not in dict1:
                    differences[key] = {"missing_in_first": dict2.get(key, "Not available")}
                elif key not in dict2:
                    differences[key] = {"missing_in_second": dict1.get(key, "Not available")}
                elif isinstance(dict1[key], str) and isinstance(dict2[key], str):
                    if dict1[key].lower() != dict2[key].lower():  # Compare case-insensitively
                        differences[key] = {"first": dict1[key], "second": dict2[key]}
                elif dict1[key] != dict2[key]:
                    differences[key] = {"first": dict1[key], "second": dict2[key]}
    return differences


def is_subset_string(str1, str2):
    # Convert both strings to lower case for case-insensitive comparison
    str1_lower = str1.lower()
    str2_lower = str2.lower()
    # Check if either string is a subset of the other
    return str1_lower in str2_lower or str2_lower in str1_lower



def calculate_sample_wise_accuracy_and_print_differencesIncludingallTables(data, ground_truth, fields):
    sample_accuracies = [0] * len(ground_truth)

    for i, gt_item in enumerate(ground_truth):
        item = data[i] if i < len(data) else None


        differences = compare_dicts_and_print(item, gt_item, fields)
        if differences is None:
            continue


        for field, diff in differences.items():
            print(f"  {field}: {diff}")

        correct_fields = 0
        field_correctness = {field: 0 for field in fields}
        composition_is_zero = False  # Initialize flag

        # Special handling for "composition" field
        if "composition" in item and "composition" in gt_item:
            comp1_amount = str(item["composition"].get("amount", "")).strip('%')
            comp2_amount = str(gt_item["composition"].get("amount", "")).strip('%')

            try:
                comp1_amount = float(comp1_amount) if comp1_amount else None
                comp2_amount = float(comp2_amount) if comp2_amount else None
            except ValueError:
                print(f"Error converting composition amounts to float: '{comp1_amount}' and '{comp2_amount}'")

            composition_is_zero = comp1_amount == 0.0 and comp2_amount == 0.0
            composition_accuracy = compare_composition(item["composition"], gt_item["composition"])
            correct_fields += composition_accuracy
            field_correctness["composition"] = composition_accuracy

        # Check other fields
        for field in fields:
            if field != "composition" and field in item and field in gt_item:
                if composition_is_zero and field in ["filler_name", "particle_surface_treatment_name"]:
                    correct_fields += 1
                    field_correctness[field] = 1
                elif isinstance(item[field], str) and isinstance(gt_item[field], str):
                    if is_subset_string(item[field], gt_item[field]):
                        correct_fields += 1
                        field_correctness[field] = 1
                elif item[field] == gt_item[field]:
                    correct_fields += 1
                    field_correctness[field] = 1

        # Calculate and store overall accuracy for the sample
        if fields:
            accuracy = correct_fields / len(fields)
            sample_accuracies[i] = accuracy

    # Calculate and return overall average accuracy
    average_accuracy = sum(sample_accuracies) / len(ground_truth)
    print(sample_accuracies)
 
    return average_accuracy, len(ground_truth)


# Function to calculate sample-wise accuracy from file paths
def calculate_sample_wise_accuracy_from_files(data_file_path, ground_truth_file_path, fields):
    data = read_json_from_file(data_file_path)
    ground_truth = read_json_from_file(ground_truth_file_path)

    if data is None or ground_truth is None:
        print("Error reading files or files are empty")
        return 0, 0

    if not data or not ground_truth:
        print("Data or ground truth is empty or not in the expected format")
        return 0, 0

    
    return calculate_sample_wise_accuracy_and_print_differencesIncludingallTables(data, ground_truth, fields)


# Function to find ground truth and files
def find_ground_truth_and_predicted_files(base_path, folder_name):
    matched_pairs = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if 'groundtruth' in root and file.endswith('.txt'):
                ground_truth_file = os.path.join(root, file)
                unique_identifier = file[:6]
                folder = os.path.join(root.replace('groundtruth', folder_name))
                if os.path.isdir(folder):
                    for file in os.listdir(folder):
                        if file.startswith(unique_identifier) and file.endswith('.txt'):
                            matched_file = os.path.join(folder, file)
                            matched_pairs.append((ground_truth_file, matched_file))
                            break
    return matched_pairs


def process_matched_pairs(matched_pairs, fields_to_check):
    total_accuracy_sum = 0
    total_samples_count = 0
    folder_count = 0  # Counter for folders
    Table_Accuracies=[]
    for ground_truth_file, predicted_file in matched_pairs:
        print(f"Processing pair: {ground_truth_file} and {predicted_file}")
        average_sample_accuracy, sample_count = calculate_sample_wise_accuracy_from_files(predicted_file, ground_truth_file, fields_to_check)
        if sample_count > 0:
            Table_Accuracies.append(average_sample_accuracy)
            total_accuracy_sum += average_sample_accuracy * sample_count
            total_samples_count += sample_count
            folder_count += 1  # Increment the folder counter
            print('folder count', folder_count)
            print(f"Average Sample Accuracy for Pair: {average_sample_accuracy}")
    print("Table_Accuracies",Table_Accuracies,"len",len(Table_Accuracies))
    print(np.mean([val for val in Table_Accuracies if val != None]))
    print("CI",get_mean_and_ci_h([val for val in Table_Accuracies if val != None])[1])
   
    print(f"Number of folders processed: {folder_count}")  # Print the total number of folders processed
    return total_accuracy_sum, total_samples_count, folder_count  # Include folder_count in the return statement

# Wrapper function to process matched pairs for different folders
def process_all_predicted_folders(base_path, folder_names, fields_to_check):
    overall_accuracy_dict = {}
    folder_count_dict = {}

    for folder_name in folder_names:
        print(f"\nProcessing Folder: {folder_name}")
        matched_pairs = find_ground_truth_and_predicted_files(base_path, folder_name)
        total_accuracy_sum, total_samples_count, folder_count = process_matched_pairs(matched_pairs, fields_to_check)
        
        overall_average_accuracy = total_accuracy_sum / total_samples_count if total_samples_count > 0 else "No valid data"
        overall_accuracy_dict[folder_name] = overall_average_accuracy
        folder_count_dict[folder_name] = folder_count

    return overall_accuracy_dict, folder_count_dict

# Base path and folder names

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
data_path = os.path.join(parent_dir, 'data')
folder_names = ["OCRoutput", "imageoutput3", "StructuredFormatoutput", "StructuredFormatwithCaptionsOutput"]

fields_to_check = ["matrix_name", "filler_name", "composition", "particle_surface_treatment_name"]


overall_accuracy_dict, folder_count_dict = process_all_predicted_folders(data_path, folder_names, fields_to_check)

