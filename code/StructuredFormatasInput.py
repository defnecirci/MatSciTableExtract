import os
import openai
import json
import sys
import re

data_string = r'''
"Identify and document detailed information about each nano and micro-composite sample listed in a provided table, using a JSON format. Detailed Instructions:
1. Sample Identification: 
o Review each sample in the table.
2. JSON Template Completion: 
o For each sample, fill out the following JSON template:

{
    \"sample_id\": [Sample ID Number],
    \"matrix_name\": [Matrix Name],
    \"filler_name\": [Filler Name],
    \"filler description\": [Filler Description], 
    \"composition\": {
        \"amount\": [Amount of Filler],
        \"type\": [Type of Composition]
    },
    \"particle_surface_treatment_name\": [Particle Surface Treatment Name],
    \"properties\": {
        [Property Name 1]: {
            \"value\": [Value],
            \"unit\": [Unit],
            \"conditions\": [
                {\"type\": [Condition Type], \"value\": [Condition Value], \"unit\": [Condition Unit]}
            ]
        },
        [Property Name 2]: {
            \"value\": [Value],
            \"unit\": [Unit],
            \"conditions\": [
                {\"type\": [Condition Type], \"value\": [Condition Value], \"unit\": [Condition Unit]}
            ]
        }
        // Add more properties as needed
    }
}

Data Entry Guidelines:
o Matrix Name: Enter the matrix's material name. Exclude any descriptors related to size or treatment.
o Filler Name: Enter only the chemical name of the filler. Exclude descriptors like nano/micro, treated/non-treated, and size.
o Filler Description: Indicate whether the filler is nano or micro. If not specified, use 'not specified'.
o Composition: Include the filler's amount (eg: 3%) and type (vol or wt or not specified). If no filler is present, enter 'none' for filler name and '0.0%' for composition. If there are reported in both of the types, just write the value and type of one of them.
o Particle Surface Treatment Name: enter chemical treatment name if known, “treated” if particles are treated but name is unknown, “untreated” if no treatment is applied, “not specified” if treatment status is unknown.
o Properties: Document all properties listed for each sample. Use full names for properties instead of abbreviations. Include value, unit, and any conditions specified. Exclude the conditions from the property name. Ignore the deviations if reported.
o If any information is missing in the table, use 'not specified' in the JSON.
Please extract all relevant information from the table and generate a complete JSON output, encompassing each nano and micro composite sample detailed in the provided table. Do not put any comments in the JSON output.
Here is an example:
[{"sample_id": 1, "matrix_name": "PP", "filler_name": "none", "filler_description": "nano”, "composition": {"amount": "0.0%", "type": "not specified"}, "particle_surface_treatment_name": "not specified", "properties": {"young's modulus": {"value": 910, "unit": "Mpa"}, "yield strength": {"value": 28, "unit": "Mpa"}, "elongation at break": {"value": 810, "unit": "%"}, "absorbed energy per thickness": {"value": 3.09, "unit": "J/cm"}, "crystallization temp": {"value": 390, "unit": "K", "conditions":[{"type": "cooling speed", "value": "-10", "unit": "K/min"}]}, "half Life of Crystallization": {"value": "120", "unit": "min", "conditions":[{"type": "temperature", "value": -413, "unit": "K"}]}}}, {"sample_id": 2, "matrix_name": "PP", "filler_name": "graphite", "filler_description": "not specified", "composition": {"amount": "2%", "type": "wt"}, "particle_surface_treatment_name": "not specified", "properties": {"young's modulus": {"value": 1300, "unit": "Mpa"}, "yield strength": {"value": "N/A", "unit": "Mpa"}, "elongation at break": {"value": 8, "unit": "%"}, "absorbed energy per thickness": {"value": 0.84, "unit": "J/cm"}, "crystallization temperature": {"value": 402, "unit": "K", "conditions":[{"type": "cooling speed", "value": -10, "unit": "K/min"}]}, "half Life of Crystallization": {"value": 9.5, "unit": "min", "conditions":[{"type": "temperature", "value": -413, "unit": "K"}]}}}]'''


def process_text_file(file_path):
    # Read the content of the text file
    with open(file_path, 'r') as file:
        text_content = file.read()
    print(f"Processing file: {file_path}")
    print(text_content)

    # Construct the API request
    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        temperature=0,
        messages=[
            {"role": "system", "content": "You extract information from documents and return json objects"},
            {"role": "user", "content": data_string +text_content}
        ]
    )

    # Extract the output
    output = response["choices"][0]["message"]["content"]
    print(output)

    # Finding the start and end indices of the JSON array
    start_index = output.find("[")
    end_index = output.rfind("]") + 1

    # Extracting the JSON array from the output
    json_array_string = output[start_index:end_index]
    json_array_string_no_comments = re.sub(r'//.*', '', json_array_string)
    # Parse and return JSON data
    try:
        json_data = json.loads(json_array_string_no_comments)
    except json.JSONDecodeError as e:
        json_data = f"Error parsing JSON: {e}"

    return json_data

def main():
    api_key = "api_key"
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)
    data_path = os.path.join(parent_dir, 'data')
    request_interval = 2
    log_file_path = os.path.join(current_dir, "imagelogfile.txt")
    #log_file_path = os.path.join(current_dir, "SFwithCaptionslogfile.txt")
    folder_name = "StructuredFormat"
    #folder_name = "StructuredFormatwithCaptions"
    output_folder_name = "StructuredFormatoutput"
    #output_folder_name = "StructuredFormatwithCaptionsoutput"

    # Set the OpenAI API key
    openai.api_key = api_key

    
    
    with open(log_file_path, "w") as log_file:
        # Redirect standard output to the log file
        sys.stdout = log_file
        # Process each text file in the SF folder of each subfolder
        for folder_name in os.listdir(data_path):
            subfolder_path = os.path.join(data_path, folder_name)
            if os.path.isdir(subfolder_path):  # Check if it's a directory
                print(f"Processing subfolder: {subfolder_path}")
                folder_path = os.path.join(subfolder_path, folder_name)
                output_folder_path = os.path.join(subfolder_path, output_folder_name)

                # Create the output folder if it doesn't exist
                if not os.path.exists(output_folder_path):
                    os.makedirs(output_folder_path)
                    print(f"Created output folder: {output_folder_path}")

                if os.path.isdir(folder_path):  # Check if the SF folder exists
                    print(f"Processing SF folder: {folder_path}")
                    for file_name in os.listdir(folder_path):
                        if file_name.endswith(".csv"):  # Process .csv files
                            file_path = os.path.join(folder_path, file_name)
                            json_data = process_text_file(file_path)

                            # Create an output file in the output folder
                            if json_data:
                                output_file_name = f"{os.path.splitext(file_name)[0]}_output.txt"
                                output_file_path = os.path.join(output_folder_path, output_file_name)
                                with open(output_file_path, "w") as file:
                                    json_string = json.dumps(json_data, indent=4)
                                    file.write(json_string)
                                print(f"Output written to: {output_file_path}")
            
        sys.stdout = sys.__stdout__
if __name__ == "__main__":
    main()
