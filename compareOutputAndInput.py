import pandas as pd
import glob
import os

def process_files(input_directory, test_file, output_directory):
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Combine all CSV files from the input directory
    file_pattern = os.path.join(input_directory, "*.csv")
    csv_files = glob.glob(file_pattern)

    if not csv_files:
        print("No CSV files found in the input directory.")
        return

    # Read and combine all input CSV files
    combined_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    # Save the combined output
    combined_output_path = os.path.join(output_directory, "combined_output.csv")
    combined_df.to_csv(combined_output_path, index=False)

    # Filter rows with Score >= 9
    df_high_score = combined_df[combined_df['Score'] >= 9]

    # Save high score output
    high_score_output_path = os.path.join(output_directory, "combined_high_output_v1.csv")
    df_high_score.to_csv(high_score_output_path, index=False)

    # Define language categories
    language_categories = {
        'EU': ['basque'],  # Add all EU language codes here
        'IT': ['italian'],
        'ES': ['spanish'],
        'EN': ['english']
    }

    # Create separate DataFrames for each category
    for category, languages in language_categories.items():
        df_category = df_high_score[df_high_score['Language'].isin(languages)]
        category_output_path = os.path.join(output_directory, f"combined_high_output_v1_{category}.csv")
        df_category.to_csv(category_output_path, index=False)

    # Optionally, handle non-EU languages if needed
    eu_languages = language_categories['EU']
    df_non_eu = df_high_score[~df_high_score['Language'].isin(eu_languages)]
    non_eu_output_path = os.path.join(output_directory, "combined_high_output_v1_non_EU.csv")
    df_non_eu.to_csv(non_eu_output_path, index=False)

    # Load test file to find missing entries
    df_test = pd.read_csv(test_file)

    # Check for missing IDs
    combined_high_ids = set(df_high_score['ID'])
    test_ids = set(df_test['ID'])
    missing_ids = test_ids - combined_high_ids

    if missing_ids:
        # Extract missing rows from the test file
        missing_rows_df = df_test[df_test['ID'].isin(missing_ids)]

        # Save missing rows by language
        for language in missing_rows_df['LANG'].unique():
            language_df = missing_rows_df[missing_rows_df['LANG'] == language]
            missing_output_path = os.path.join(output_directory, f"MISSING_test_{language}.csv")
            language_df.to_csv(missing_output_path, index=False)
    else:
        print("No missing IDs found.")

    print("Processing complete. Outputs saved to:", output_directory)


# Example usage
input_directory = "./testingDataOutput"
test_file = "./testingData/test.csv"
output_directory = "./processedOutput"

process_files(input_directory, test_file, output_directory)
