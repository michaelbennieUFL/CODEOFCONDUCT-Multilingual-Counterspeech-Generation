import pandas as pd
import glob
import os


def sort_responses(input_files, output_dir):
    # Initialize empty lists for each ranking
    ranked_dfs = {1: [], 2: [], 3: [], 4: []}

    # Process each input file
    for file in input_files:
        # Read the CSV file
        df = pd.read_csv(file)

        # Sort responses into respective rankings
        for rank in range(1, 5):
            # Filter rows by ranking
            ranked_df = df[df['ranking'] == rank]

            # Create dataframe with required columns
            output_df = pd.DataFrame({
                'ID': ranked_df['ID'],
                'KN': '',  # Empty column as per instructions
                'KN_CN': ranked_df['Response']
            })

            # Add to the corresponding rank list
            if not output_df.empty:
                ranked_dfs[rank].append(output_df)

    # Combine and save results for each ranking
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
    for rank in range(1, 5):
        # Combine all dataframes for this ranking
        if ranked_dfs[rank]:  # Check if there are any dataframes to combine
            combined_df = pd.concat(ranked_dfs[rank], ignore_index=True)

            # Save to CSV file with the required filename in the output directory
            output_filename = os.path.join(output_dir, f'CODEOFCONDUCT-run{rank}-predictions.csv')
            combined_df.to_csv(output_filename, index=False)
            print(f"Created {output_filename}")


def sort_responses_for_local_eval(input_files, output_dir="generated"):
    # Initialize empty lists for each ranking
    ranked_dfs = {1: [], 2: [], 3: [], 4: []}

    # Process each input file
    for file in input_files:
        # Read the CSV file
        df = pd.read_csv(file)

        # Sort responses into respective rankings
        for rank in range(1, 5):
            # Filter rows by ranking
            ranked_df = df[df['ranking'] == rank]

            # Create dataframe with the required columns
            output_df = pd.DataFrame({
                'HS': ranked_df['HateSpeech'],
                'Label': ranked_df['ID'],
                'generated': ranked_df['Response']
            })

            # Add to the corresponding rank list
            if not output_df.empty:
                ranked_dfs[rank].append(output_df)

    # Combine and save results for each ranking
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
    for rank in range(1, 5):
        # Combine all dataframes for this ranking
        if ranked_dfs[rank]:  # Check if there are any dataframes to combine
            combined_df = pd.concat(ranked_dfs[rank], ignore_index=True)

            # Save to CSV file with the required filename in the output directory
            output_filename = os.path.join(output_dir, f'CODEOFCONDUCT-run{rank}-predictions.csv')
            combined_df.to_csv(output_filename, index=False)
            print(f"Created {output_filename}")


# Execute the script
if __name__ == "__main__":
    input_dir = "optimizedResultsByLanguage"  # Directory with input files
    output_dir = "outputs"  # Directory to save output files

    # Get all CSV files in the specified input directory
    input_files = glob.glob(os.path.join(input_dir, "*.csv"))

    if input_files:
        print(f"Processing {len(input_files)} input files from '{input_dir}'...")
        sort_responses(input_files, output_dir)

        input_files = glob.glob(os.path.join(input_dir, "*.csv"))
        sort_responses_for_local_eval(input_files)
    else:
        print(f"No input CSV files found in '{input_dir}'.")
