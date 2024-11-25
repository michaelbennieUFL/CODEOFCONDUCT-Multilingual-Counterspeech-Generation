import math
import os
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm

# Assuming findBestCounterSpeech and related dependencies are imported or defined elsewhere
from advercialModel.GreedyAlgorithm import findBestCounterSpeech

def map_language_code(lang_code: str) -> str:
    """
    Maps a 2-digit language code to the full language name.

    Args:
    lang_code (str): 2-digit language code.

    Returns:
    str: Full language name.
    """
    language_map = {
        "EN": "english",
        "ES": "spanish",
        "IT": "italian",
        "EU": "basque"
    }
    return language_map.get(lang_code, "unknown")

def average_high_score_from_csv(
    input_csv_path: str,
    iterations: int,
    heatIncrement: float,
    numAnswersToGenerateForEachLoop: int,
    sampleSize: int = 10
) -> Tuple[int, float, int, float]:
    """
    Calculates the average of the highest scores outputted for each line in the CSV.

    Args:
    input_csv_path (str): Path to the input CSV file.
    iterations (int): Number of iterations for simulated annealing.
    heatIncrement (float): Heat increment value.
    numAnswersToGenerateForEachLoop (int): Number of answers to generate in each loop.
    sampleSize (int): Sample size for words.

    Returns:
    Tuple[int, float, int, float]: (iterations, heatIncrement, numAnswersToGenerateForEachLoop, average high score)
    """
    input_data = pd.read_csv(input_csv_path)
    high_scores = []

    # Iterate over each row and generate counter-speech
    for _, row in tqdm(input_data.iterrows(), total=len(input_data), desc="Processing input CSV"):
        ID = row['ID']
        hateSpeech = row['HS']
        KN = row['KN']
        language_code = row['LANG']
        language = map_language_code(language_code)

        # Generate counter-speech for each entry
        responses = findBestCounterSpeech(
            ID, hateSpeech, KN, language,
            sampleSize=sampleSize,
            iterations=iterations,
            numAICallsPerAILoop=numAnswersToGenerateForEachLoop,
            generateAiAnswersPeriod=1,
            heatIncrement=heatIncrement
        )

        # responses is a list of [ID, KN, response, score, language, hateSpeech]
        # Extract the highest score
        if responses:
            # Get the maximum score from responses
            scores = [item[3] for item in responses]  # item[3] is the score
            max_score = max(scores)
            high_scores.append(max_score)
        else:
            # If no responses were generated, append zero
            high_scores.append(0)

    # Calculate the average of the highest scores
    if high_scores:
        average_high_score = sum(high_scores) / len(high_scores)
    else:
        average_high_score = 0

    return (iterations, heatIncrement, numAnswersToGenerateForEachLoop, average_high_score)

def run_parameter_grid(
    input_csv_path: str,
    output_csv_path: str,
    iterations_list: List[int],
    numAnswersToGenerateForEachLoop_list: List[int],
    heatIncrement: float = 0.0,
    sampleSize: int = 10
) -> None:
    """
    Runs the average_high_score_from_csv function over ranges of parameters and saves the results.
    Now saves the results to the CSV file during each iteration.

    Args:
        input_csv_path (str): Path to the input CSV file.
        output_csv_path (str): Path to save the output CSV file with results.
        iterations_list (List[int]): List of iteration counts to test.
        numAnswersToGenerateForEachLoop_list (List[int]): List of numAnswersToGenerateForEachLoop values to test.
        heatIncrement (float): Heat increment value.
        sampleSize (int): Sample size for words.

    Returns:
        None
    """
    # Define the columns for the CSV
    columns = ['iterations', 'heatIncrement', 'numAnswersToGenerateForEachLoop', 'average_high_score']

    # Check if the output CSV file exists; if not, create it with headers
    if not os.path.exists(output_csv_path):
        pd.DataFrame(columns=columns).to_csv(output_csv_path, index=False)

    # Iterate over all combinations of parameters
    for iterations in iterations_list:
        for numAnswersToGenerateForEachLoop in numAnswersToGenerateForEachLoop_list:
            print(f"Running with iterations={iterations}, numAnswersToGenerateForEachLoop={numAnswersToGenerateForEachLoop}")

            # Call the function to get average high score
            _, _, _, average_high_score = average_high_score_from_csv(
                input_csv_path,
                iterations,
                heatIncrement,
                numAnswersToGenerateForEachLoop,
                sampleSize
            )

            # Prepare the result as a DataFrame
            result_df = pd.DataFrame([{
                'iterations': iterations,
                'heatIncrement': heatIncrement,
                'numAnswersToGenerateForEachLoop': numAnswersToGenerateForEachLoop,
                'average_high_score': average_high_score
            }])

            # Append the result to the CSV file
            result_df.to_csv(output_csv_path, mode='a', header=False, index=False)


if __name__ == "__main__":
    input_csv_path = "testingData/small_testset.csv"
    output_csv_path = "parameter_grid_results.csv"
    iterations_list = [2,4,6,8]
    numAnswersToGenerateForEachLoop_list = [2, 4, 6]
    heatIncrement = 0.0
    sampleSize = 7

    run_parameter_grid(
        input_csv_path,
        output_csv_path,
        iterations_list,
        numAnswersToGenerateForEachLoop_list,
        heatIncrement,
        sampleSize
    )
