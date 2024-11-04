from advercialModel.JudgeLMinterface import JudgeLMEvaluator
from itertools import combinations, combinations_with_replacement
import pandas as pd
from tqdm import tqdm


def prettyPrintResults(results: list[tuple[str, float]],response_length=20):
    # ANSI color codes
    green = "\033[92m"  # Green for tags
    red = "\033[91m"  # Red for score values
    reset = "\033[0m"  # Reset color to default

    for response, score in results:
        # Truncate and add "..." if longer than 15 characters
        if len(response) > response_length:
            formatted_response = response[:response_length-3] + "..."
        else:
            # Pad with spaces to make it 15 characters
            formatted_response = response.ljust(response_length)

        # Print formatted output with colors
        print(f"{green}答案:{reset} {formatted_response} | {green}平均分:{reset} {red}{score}{reset}")



def roundRobinCalculator(hateSpeech: str, previousAnswers: list[str]) -> list[str,float]:
    """
    Performs round-robin tournament scoring of answers using JudgeLM evaluator.
    
    Args:
        hateSpeech (str): The hate speech content to evaluate against
        previousAnswers (list): List of answer strings to evaluate
        
    Returns:
        list: List of [answer, average_score] pairs
    """
   
    evaluator = JudgeLMEvaluator()

    answer_pairs = list(combinations_with_replacement(previousAnswers, 2))

    scores={}
    # For each pair, create a question format that JudgeLM expects
    for answer1, answer2 in answer_pairs:
        # Create question format
        question = {
            "question_body": f"Provide a brief counter-narrative in response to the following hate speech: {hateSpeech}. It is very important for the counter-narrative to be relevant to the hate speech.",
            "answer1_body": answer1,
            "answer2_body": answer2
        }
        
        # Get normal and reversed evaluations
        normal_results = evaluator.get_model_answers([question], if_reverse_answers=False)[0]
        reversed_results = evaluator.get_model_answers([question], if_reverse_answers=True)[0]
        
        # Add the scores from both evaluations
        score1 = normal_results["output1"] + reversed_results["output1"]
        score2 = normal_results["output2"] + reversed_results["output2"]
        
        # Update scores and match counts
        scores[answer1] = scores.get(answer1, 0) + score1
        #Stop the double-scoring of answers with themselves
        if answer1 != answer2:
            scores[answer2] = scores.get(answer2, 0) + score2


    average_scores = []
    numberOfMatches=2*len(previousAnswers)
    for answer in previousAnswers:
        avg_score = scores[answer] / numberOfMatches
        average_scores.append([answer, avg_score])
    
    # Sort by average score in descending order
    average_scores.sort(key=lambda x: x[1], reverse=True)
    
    return average_scores

def process_csv_file(input_file: str, output_file: str):
    """
    Process a CSV file containing hate speech responses and evaluate them using round-robin tournament.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Group by ID to get all responses for each hate speech
    grouped = df.groupby('ID')
    
    # Initialize lists for results
    all_ids = []
    all_responses = []
    all_scores = []
    all_hate_speeches = []  # Added to keep track of hate speeches
    
    # Process each group
    for id_val, group in tqdm(grouped, desc="Processing groups"):
        # Get list of responses and hate speech for this ID
        responses = group['Response'].tolist()
        hate_speech = group['HateSpeech'].iloc[0]  # Get hate speech for this ID
        
        # Calculate scores using round robin tournament
        scored_responses = roundRobinCalculator(str(id_val), hate_speech, responses)
        
        # Add results to lists
        for response, score in scored_responses:
            all_ids.append(id_val)
            all_responses.append(response)
            all_scores.append(score)
            all_hate_speeches.append(hate_speech)
    
    # Create output dataframe
    output_df = pd.DataFrame({
        'ID': all_ids,
        'HateSpeech': all_hate_speeches,
        'Response': all_responses,
        'Average_Score': all_scores
    })
    
    # Sort by ID and Score
    output_df = output_df.sort_values(['ID', 'Average_Score'], ascending=[True, False])
    
    # Save to CSV
    output_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    # Example usage
    input_file = "./testingDataOutput/output_counter_speech_EN_split_1.csv"
    output_file = "./testingDataOutput/scored_counter_speech_EN_split_1.csv"
    
    #process_csv_file(input_file, output_file)
    previous_answers = [
        "We should respect all people!",
        "Every individual has the right to love and be loved, regardless of their sexual orientation, and promoting acceptance and inclusivity can help create a more compassionate and equitable society.",
        "Every individual has the right to love and be loved, regardless of their sexual orientation, and promoting acceptance and inclusivity can help create a more compassionate and equitable society. printed",
        "Instead of resorting to hate or deletion, let's strive to build a more equitable society where everyone has the right to love and be loved, by fostering a culture of empathy, understanding, and acceptance of all individuals, including those from the LGBTQ+ community."
    ]

    hate_speech = "Faggots make me angry, why don't we delete them?"
    prettyPrintResults(roundRobinCalculator(hate_speech,previous_answers))