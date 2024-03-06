import os
import base64
import requests
import json
import openai
import time
import sys

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

# Function to encode an image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to process an image and extract JSON data
def process_image(image_path, api_key):
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Construct the payload

    payload = {
  "model": "gpt-4-vision-preview",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          #"text": "Please analyze the provided image of a table that contains information about various nano-composite samples. Your task is to identify each nano-composite sample listed in the table and then fill out the provided JSON template for each sample. The JSON should include details such as the sample ID number, matrix name, filler name, filler description (nano or micro), composition (including amount and type), particle surface treatment name, and properties. Note that properties can vary from one article to another, so please include all the properties mentioned for each sample in the table. The JSON template for each nano-composite sample should be structured as follows:\n\n{\n    \"sample_id\": [Sample ID Number],\n    \"matrix_name\": [Matrix Name],\n    \"filler_name\": [Filler Name],\n    \"filler description\": [Filler Description],\n    \"composition\": {\n        \"amount\": [Amount of Filler],\n        \"type\": [Type of Composition]\n    },\n    \"particle_surface_treatment_name\": [Particle Surface Treatment Name],\n    \"properties\": {\n        [Property Name 1]: {\n            \"value\": [Value],\n            \"unit\": [Unit],\n            \"conditions\": [\n                {\"type\": [Condition Type], \"value\": [Condition Value], \"unit\": [Condition Unit]}\n            ]\n        },\n        [Property Name 2]: {\n            \"value\": [Value],\n            \"unit\": [Unit],\n            \"conditions\": [\n                {\"type\": [Condition Type], \"value\": [Condition Value], \"unit\": [Condition Unit]}\n            ]\n        }\n        // Add more properties as needed\n    }\n}\n\nPlease extract all relevant information from the table and generate a complete JSON output, encompassing each nano-composite sample detailed in the table. Here is an example: [ { \"sample_id\": 1, \"matrix_name\": \"PP\", \"filler_name\": null, \"filler description\":null, \"composition\": { \"amount\": null, \"type\": \"Not Specified\" }, \"particle_surface_treatment_name\": \"Not Specified\", \"properties\": { \"young's modulus\": {\"value\": 910, \"unit\": \"Mpa\"}, \"yield strength\": {\"value\": 28, \"unit\": \"Mpa\"}, \"elongation at break\": {\"value\": 810, \"unit\": \"%\"}, \"absorbed energy per thickness\": {\"value\": 3.09, \"unit\": \"J/cm\"}, \"crystallization temp\": {\"value\": 390, \"unit\": \"K\", \"conditions\":[{\"type\": \"cooling speed\", \"value\": \"-10\", \"unit\": \"K/min\"}] }, \"half Life of Crystallization\": {\"value\": \">120\", \"unit\": \"min\", \"conditions\":[{\"type\": \"temperature\", \"value\": \"-413\", \"unit\": \"K\"}] }, } }, { \"sample_id\": 2, \"matrix_name\": \"PP\", \"filler_name\": \"graphite melt extrution\", \"filler description\":null, \"composition\": { \"amount\": \"2.8\", \"type\": \"wt%\" }, \"particle_surface_treatment_name\": \"Not Specified\", \"properties\": { \"young's modulus\": {\"value\": 1300, \"unit\": \"Mpa\"}, \"yield strength\": {\"value\": \"N/A\", \"unit\": \"Mpa\"}, \"elongation at break\": {\"value\": 8, \"unit\": \"%\"}, \"absorbed energy per thickness\": {\"value\": 0.84, \"unit\": \"J/cm\"}, \"crystallization temp\": {\"value\": 402, \"unit\": \"K\", \"conditions\":[{\"type\": \"cooling speed\", \"value\": \"-10\", \"unit\": \"K/min\"}] }, \"half Life of Crystallization\": {\"value\": \">9.5\", \"unit\": \"min\", \"conditions\":[{\"type\": \"temperature\", \"value\": \"-413\", \"unit\": \"K\"}] }, } }]"},
           "text": data_string },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
          }
        }
        #,
        #{
         # "type": "image_url",
         # "image_url": {
        #    "url": f"data:image/jpeg;base64,{base64_image2}"
        #  }
       # }
      ]
    }
  ],
  "max_tokens": 3000
}
    # Sending the request to the API
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    # Check if the response is successful
    if response.status_code == 200:
        try:
            assistant_message = response.json()['choices'][0]['message']['content']
            # Extracting and returning the JSON part
            parts = assistant_message.split("```")
            print(assistant_message) 
            return parts[1] if len(parts) > 1 else None
        except (KeyError, IndexError, json.JSONDecodeError):
            print(f"Error parsing response for image {image_path}: {response.text}")
            return None
    else:
        print(f"API request failed for image {image_path}: {response.status_code} - {response.text}")
        return None


def main():
    
    api_key = "api_key"
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)
    data_path = os.path.join(parent_dir, 'data')
    request_interval = 2
    log_file_path = os.path.join(current_dir, "imagelogfile.txt")
    with open(log_file_path, "w") as log_file:
        # Redirect standard output to the log file
        sys.stdout = log_file
      # Process each subfolder in the main folder
        for folder_name in os.listdir(data_path):
            subfolder_path = os.path.join(data_path, folder_name)
            print(folder_name)
            if os.path.isdir(subfolder_path):  # Check if it's a directory
                tables_folder_path = os.path.join(subfolder_path, "tables")
                imageoutput_folder_path = os.path.join(subfolder_path, "imageoutput3")  # Path for the imageoutput folder

                # Create the imageoutput folder if it does not exist
                if not os.path.exists(imageoutput_folder_path):
                    os.makedirs(imageoutput_folder_path)

                if os.path.exists(tables_folder_path):  # Check if the 'tables' folder exists
                    for image_name in os.listdir(tables_folder_path):
                        print(image_name)
                        if image_name.endswith(".png"):  # or any other image format
                            image_path = os.path.join(tables_folder_path, image_name)
                            json_data = process_image(image_path, api_key)

                            # Create an output file for each image in the imageoutput folder
                            if json_data:
                                output_file = os.path.join(imageoutput_folder_path, f"{os.path.splitext(image_name)[0]}_output3.txt")
                                with open(output_file, "w") as file:
                                    file.write(json_data[5:])
                            time.sleep(request_interval)
        sys.stdout = sys.__stdout__
if __name__ == "__main__":
    main()  
