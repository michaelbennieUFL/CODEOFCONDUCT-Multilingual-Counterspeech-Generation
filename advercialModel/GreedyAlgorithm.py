import random
import math
from typing import List

from advercialModel.JudgeLMinterface import JudgeLMEvaluator
from advercialModel.LLMAnswerGenerator import LLMAnswerGenerator
from advercialModel.WordList import generate_frequent_word_list

evaluator=JudgeLMEvaluator()
llmAnswerGenerator=LLMAnswerGenerator()


def sample_words_from_text(text:str, sample_size:int=1000000) -> List[str]:
    """
    Sample a specified number of words from the given string 'text' list.
    """
    # Split the text by spaces to get words
    words = text.split()
    # Get unique words by converting to a set
    unique_words = list(set(words))
    # Randomly sample words from the unique words list with the given sample size
    sample = random.sample(unique_words, min(sample_size, len(unique_words)))
    return sample

def sample_words(wordList:List[str], sample_size:int)->List[str]:
    """
    Sample a specified number of words from the word list.
    """
    return random.sample(wordList, min(sample_size, len(wordList)))


def calculate_scores(words:List[str], question:str, answer:str) -> dict[str, float]:
    """
    Calculate scores for each word pair based on the question and current answer.
    """
    scores = {}

    # If the list has an odd number of words, duplicate the first word at the end
    if len(words) % 2 != 0:
        words.append(words[0])

    # Iterate over words in pairs
    for i in range(0, len(words), 2):
        word1 = words[i]
        word2 = words[i + 1]

        # Placeholder for scoring logic, using a hypothetical score_word function
        # Here, it's stored as a combined key "word1-word2" in the scores dictionary
        answer1 = answer + " " + word1
        answer2 = answer + " " + word2
        scores[answer1],scores[answer2] = score_answer2(answer1, answer2, question)

    return scores




def convert_to_probabilities(weighted_scores:dict[str, float])->dict[str, float]:
    """
    Convert weighted scores into probabilities.
    """
    total = sum(weighted_scores.values())
    if total == 0:
        return {k: 1 / len(weighted_scores) for k in weighted_scores}
    return {k: v / total for k, v in weighted_scores.items()}


import random

def probabilistic_selection(scores, num_selections):
    """
    Select a specified number of unique items based on their probabilities,
    ensuring that one of the top items is always selected first.
    If num_selections is greater than the number of items, it selects all items.
    """
    # Adjust num_selections if it exceeds the number of unique items
    num_selections = min(num_selections, len(scores))

    # Convert scores to a list for sampling and removing items
    items = list(scores.items())
    weights = list(scores.values())

    # Initialize selected items list
    selected = []

    # Find the top item based on the highest score and select it first
    max_index = weights.index(max(weights))
    top_item = items.pop(max_index)
    weights.pop(max_index)
    selected.append(top_item)

    # Adjust num_selections to account for the top item selection
    remaining_selections = num_selections - 1

    try:
        # Loop to ensure unique selections without re-sampling the same item
        for _ in range(remaining_selections):
            # Select an item based on weights
            chosen = random.choices(items, weights=weights, k=1)[0]
            selected.append(chosen)

            # Remove the selected item and its weight for future draws
            index = items.index(chosen)
            items.pop(index)
            weights.pop(index)
    except Exception as e:
        print(e)

    return dict(selected)



def generate_new_answers(answer,prompt)->List[str]:
    """
    Generate two new possible answers using a Language Model (LLM).
    """
    # Placeholder for LLM generation logic
    # Example: new_ans1, new_ans2 = llm_generate_answers(answer)
    formatted_prompt=llmAnswerGenerator.generate_formatted_prompt(prompt,answer)
    random_response = llmAnswerGenerator.generate_random_response(formatted_prompt=formatted_prompt)
    parsed_answers_random = llmAnswerGenerator.parse_answers(random_response)
    return parsed_answers_random



def score_answer2(answer1:str,answer2:str, question:str)->(int,int):
    """
    Score a single answer based on the question.
    """
    # Placeholder for answer scoring logic
    # Example: return some_answer_scoring_function(answer, question)
    questions_test = [
        {
         "question_body": question,
         "answer1_body": answer1,
         "answer2_body": answer2,
         },
    ]

    resultsNormal=evaluator.get_model_answers(questions_test)[0]
    resultsInverted = evaluator.get_model_answers(questions_test,if_reverse_answers=True)[0]
    try:
        averageOutput1=(resultsNormal["output1"]+resultsInverted["output1"])/2
        averageOutput2 = (resultsNormal["output2"] + resultsInverted["output2"]) / 2
    except Exception:
        averageOutput1,averageOutput2=0,0
    return averageOutput1,averageOutput2


def generateExponentialWeightedScores(answers:dict[str,float],exponentBase:float) -> dict[str,float]:
    weighted_scores = {ans: exponentBase ** score for ans, score in answers.items()}
    return weighted_scores

def simulatedAnnealingSearchStep(question, answer, wordList,
                                 sampleSize=10,
                                 sampleSizeScoreRatio=1,
                                 numAnswersToGenerate=3,
                                 scoreWeighting=100,
                                 generateAIAnswers=False,
                                 ):
    # Step 1: Sample 100 or so words
    sampled_words = sample_words(wordList, sampleSize)
    sampled_words.append("")

    # Step 2: Calculate scores of each answer
    answer_scores = calculate_scores(sampled_words, question, answer)


    # Step 3: Weight scores with exponential function 10^(score) so that higher scores are exponentially more likely
    weighted_scores=generateExponentialWeightedScores(answer_scores, scoreWeighting)

    # Step 4: Select 5 answer-score pairs probabilistically (I think this is what simulated annealing is sposto be.)
    selected_answers = probabilistic_selection(weighted_scores, numAnswersToGenerate)

    if not generateAIAnswers:
        return selected_answers

    # Step 5: For each selected answer, generate  other possible answers with LLM and score them
    generated_answers = []
    for ans in selected_answers:
        generated_answers+=generate_new_answers(ans,prompt=question)

    generated_answers_scores =calculate_scores(generated_answers, question, "")

    # Step 6: Score the 20 or so new answers and weight them exponentially
    generated_answers_scores_weighted = generateExponentialWeightedScores(generated_answers_scores, scoreWeighting)

    # Step 7: Convert all the answer scores into probabilities and choose 4 answers
    combined_dict = selected_answers | generated_answers_scores_weighted
    final_selected_answers = probabilistic_selection(combined_dict, numAnswersToGenerate)

    return final_selected_answers

def print_words(exponential_scores: dict[str, float], base: float) -> None:
    """
    Print answers with their exponential scores converted back to regular scores.
    Displays only the first 20 characters of each answer followed by ellipses if longer than 20 characters.
    Ensures the commas align by using consistent formatting.
    """
    # Determine maximum length for answer display (20 characters + ellipses space)
    number_of_letters = 20
    max_display_length = number_of_letters+3
    for answer, exp_score in exponential_scores.items():
        # Convert the exponential score back to a regular score using the specified base
        regular_score = math.log(exp_score, base) if exp_score > 0 else float('-inf')

        # Truncate answer to the first 20 characters, adding ellipses if longer
        display_answer = (answer[:number_of_letters] + "...") if len(answer) > number_of_letters else answer
        # Format answer to fixed length for alignment
        formatted_answer = f"{display_answer:<{max_display_length}}"

        print(f"Answer: {formatted_answer}, Score: {regular_score:.2f}")

def simulatedAnnealing(question, answer, wordList,
                       targetScore=10, startingPossibleAffixes={"":0.1},
                       maxIterations=10,sampleSize=20,
                       numAnswersToGenerateForEachLoop=4):
    currentPossibleAffixes=startingPossibleAffixes
    scoreWeighting=sampleSize


    generateAiAnswersPeriod = 10

    #5*10*10=500*100=50,000*4=200,000

    for i in range(maxIterations):
        newPossibleAffixes=currentPossibleAffixes
        for ans,score in currentPossibleAffixes.items():
            if (score >= scoreWeighting**targetScore):
                return ans,currentPossibleAffixes,"Success"
            aiAnswers:bool=not (i %generateAiAnswersPeriod)
            newPossibleAffixes = newPossibleAffixes | simulatedAnnealingSearchStep(question,
                                                                                   ans,
                                                                                   wordList,
                                                                                   sampleSize=sampleSize,
                                                                                   scoreWeighting=scoreWeighting,
                                                                                   generateAIAnswers=aiAnswers)
            print(newPossibleAffixes)
            print_words(newPossibleAffixes,scoreWeighting)
        currentPossibleAffixes = probabilistic_selection(newPossibleAffixes, numAnswersToGenerateForEachLoop)
        print("FINISHED ITERATION #{}".format(i))
        print_words(currentPossibleAffixes, scoreWeighting)
        print("\n--------------------------------\n\n")
    return currentPossibleAffixes[0][0],currentPossibleAffixes,"Failure"

if __name__ =="__main__":
    hate_speech="Jewish Supremacists are too focused on world domination. They want to rule the world."
    prompt_example   = "Provide a brief counter-narrative in response to the following hate speech: "+hate_speech+" It is very important for the counter-narrative to be relevant to the hate speech."
    sentence_example = "Of course Jews are focused on 'world domination', even \"galaxy domination\". But so are Sith Order, Sauron etc. It is not accurate or fair to label any group."
    word_list=(generate_frequent_word_list(min_length=5)
               +sample_words_from_text(hate_speech)*500)
    result=simulatedAnnealing(question=prompt_example,answer=sentence_example,wordList=word_list)
    print(result)

    #use this regex to find perfect scores Output[12]:\s*10