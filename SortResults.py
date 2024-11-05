import pandas as pd
import glob
import os

def sort_responses(input_files):
    # Initialize empty dataframes for each ranking
    ranked_dfs = {1: [], 2: [], 3: []}
    
    # Process each input file
    for file in input_files:
        # Read the CSV file
        df = pd.read_csv(file)
        
        # Sort responses into respective rankings
        for rank in range(1, 4):
            # Filter rows by ranking
            ranked_df = df[df['ranking'] == rank]
            
            # Create dataframe with required columns
            output_df = pd.DataFrame({
                'ID': ranked_df['ID'],
                'KN': '',  # Empty column
                'KN_CN': ranked_df['Response']
            })
            
            # Add to the corresponding rank list
            ranked_dfs[rank].append(output_df)
    
    # Combine and save results for each ranking
    for rank in range(1, 4):
        # Combine all dataframes for this ranking
        if ranked_dfs[rank]:  # Check if there are any dataframes to combine
            combined_df = pd.concat(ranked_dfs[rank], ignore_index=True)
            
            # Save to CSV file
            output_filename = f'rank_{rank}_responses.csv'
            combined_df.to_csv(output_filename, index=False)
            print(f"Created {output_filename}")

# Execute the script
if __name__ == "__main__":
    # Get all CSV files in the current directory
    input_files = glob.glob("*.csv")
    
    # Filter out the output files if they exist
    input_files = [f for f in input_files if not f.startswith('rank_')]
    
    if input_files:
        print(f"Processing {len(input_files)} input files...")
        sort_responses(input_files)
    else:
        print("No input CSV files found in the current directory.") 
