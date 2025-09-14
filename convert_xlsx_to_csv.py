import pandas as pd
import os

def convert_xlsx_to_csv(data_folder="data"):
    """
    Converts all .xlsx files in the specified data_folder to .csv files.
    """
    full_data_path = os.path.join("/Users/danial/PythonProject", data_folder)
    
    for filename in os.listdir(full_data_path):
        if filename.endswith(".xlsx"):
            xlsx_path = os.path.join(full_data_path, filename)
            csv_filename = filename.replace(".xlsx", ".csv")
            csv_path = os.path.join(full_data_path, csv_filename)

            print(f"Converting {xlsx_path} to {csv_path}...")
            try:
                df = pd.read_excel(xlsx_path)
                df.to_csv(csv_path, index=False, encoding='utf-8')
                print(f"Successfully converted {filename} to {csv_filename}")
            except Exception as e:
                print(f"Error converting {filename}: {e}")

if __name__ == "__main__":
    convert_xlsx_to_csv()