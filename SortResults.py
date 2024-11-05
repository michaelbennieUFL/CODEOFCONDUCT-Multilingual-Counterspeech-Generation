import pandas as pd

def sort_responses(input_file):
    
    df = pd.read_csv(input_file)
    
    # Create separate dataframes for each ranking (1, 2, 3)
    for rank in range(1, 4):
        # Filter rows by ranking
        ranked_df = df[df['ranking'] == rank]
        
        # Create new dataframe with required columns
        output_df = pd.DataFrame({
            'ID': ranked_df['ID'],
            'KN': '',  # Empty column
            'KN_CN': ranked_df['Response']
        })
        
        # Save to CSV file
        output_filename = f'rank_{rank}_responses.csv'
        output_df.to_csv(output_filename, index=False)
        print(f"Created {output_filename}")

# Execute the script
if __name__ == "__main__":
    input_file = "ES_Final_Results.csv"
    sort_responses(input_file) 
