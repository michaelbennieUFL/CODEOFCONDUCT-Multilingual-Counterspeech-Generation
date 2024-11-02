import math
from typing import List

import pandas as pd
from tqdm import tqdm

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

def generate_frequent_word_list(language: str) -> List[str]:
    """
    Generate a mock frequent word list based on language.

    Args:
    language (str): Language to generate the word list for.

    Returns:
    List[str]: A list of frequent words in the specified language.
    """
    mock_words = {
        "english": ["freedom", "rights", "democracy", "community"],
        "spanish": ["libertad", "derechos", "democracia", "comunidad"],
        "italian": ["libertà", "diritti", "democrazia", "comunità"],
        "basque": ["askatasuna", "eskubideak", "demokrazia", "komunitatea"]
    }
    return mock_words.get(language, [])



def GenerateAnswersFromCSV(input_csv_path: str, output_csv_path: str, sampleSize=25, iterations=13,numAICallsPerAILoop=5):
    """
    Generate counter-speech responses from an input CSV file containing hate speech.

    Args:
    input_csv_path (str): Path to the input CSV file.
    output_csv_path (str): Path to save the output CSV file with counter-speech.
    """
    # Read input data
    input_data = pd.read_csv(input_csv_path)

    # Prepare output data list
    output_data = []

    # Iterate over each row and generate counter-speech with tqdm for progress tracking
    for _, row in tqdm(input_data.iterrows(), total=len(input_data), desc="Generating counter-speech"):
        ID = row['ID']
        hateSpeech = row['HS']
        KN = row['KN']
        language_code = row['LANG']
        language = map_language_code(language_code)

        # Generate counter-speech for each entry
        responses = findBestCounterSpeech(ID, hateSpeech, KN, language,
                                          sampleSize=sampleSize,
                                          iterations=iterations,
                                          numAICallsPerAILoop=numAICallsPerAILoop,
                                          generateAiAnswersPeriod=iterations//4)
        output_data.extend(responses)

    # Convert the output list to a DataFrame and save as CSV
    output_df = pd.DataFrame(output_data, columns=["ID", "KN", "Response", "Score", "Language", "HateSpeech"])
    output_df.to_csv(output_csv_path, index=False)

if __name__ =="__main__":
    # Define paths for testing
    input_csv_path = './testingData/test_IT.csv'
    output_csv_path = './TestingDataOutputAnswers/output_counter_speech_IT.csv'

    # Run the function on the test data
    GenerateAnswersFromCSV(input_csv_path, output_csv_path,sampleSize=7,iterations=31)


