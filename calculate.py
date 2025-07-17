import re
import json
import os
import argparse

def normalize_answer(s):
    """
    Normalizes a string by removing common formatting and units.

    This function performs the following operations:
    1. Removes LaTeX commands like \frac, \text, \$, etc.
    2. Removes commas from numbers (e.g., "1,000" -> "1000" but not from lists like "1, 2").
    3. Removes trailing dollar signs, percentages, and common units.
    4. Strips leading/trailing whitespace.
    """
    s = str(s).strip()
    # Handle fractions
    s = re.sub(r"\\frac\{(.*?)\}\{(.*?)\}", r"\1/\2", s)
    s = re.sub(r"\\dfrac\{(.*?)\}\{(.*?)\}", r"\1/\2", s)
    
    # Remove LaTeX and common text artifacts
    s = re.sub(r"\\text\{.*?\}", "", s)
    # Remove dollar signs, but not commas that might separate list items
    s = s.replace("$", "")
    
    # Remove units and percentages
    s = s.replace("%", "")
    
    # Remove "dozens" and other words attached to numbers
    s = re.sub(r"\s*(dozens?|eggs?|dollars?)\b", "", s, flags=re.IGNORECASE)

    s = s.strip()
    s = s.replace(" ", "")
    s = s.replace("\\", "")
    s = s.replace("\\\\", "")
    return s

def extract_final_answer(model_output):
    """
    Extracts the final numerical answer from a model's output.

    It tries to find the answer in a specific order of patterns:
    1. Inside a LaTeX \boxed{} environment.
    2. After the '####' delimiter.
    3. After "The final answer is".
    4. After "The answer is".
    5. The last numerical value in the string as a fallback.
    """
    # 1. Check for \boxed{}
    # The regex looks for \boxed{...} and captures the content inside.
    boxed_match = re.search(r"\\boxed{(?P<answer>.*?)}", model_output, re.DOTALL)
    if boxed_match:
        return normalize_answer(boxed_match.group("answer"))

    # 2. Check for '####' delimiter (common in GSM8K)
    # This pattern is often used to mark the final answer clearly.
    final_answer_match = re.search(r"####\s*(.*)", model_output)
    if final_answer_match:
        return normalize_answer(final_answer_match.group(1).strip())

    # 3. Check for "The final answer is"
    # The regex looks for the phrase (case-insensitive) and captures the text following it.
    final_answer_match = re.search(r"(?i)the final answer is:?\s*([-+]?\d*\.?\d+)", model_output)
    if final_answer_match:
        return normalize_answer(final_answer_match.group(1))
        
    # 4. Check for "the answer is"
    # A more conversational but common pattern.
    answer_is_match = re.search(r"(?i)the answer is:?\s*([-+]?\d*\.?\d+)", model_output)
    if answer_is_match:
        return normalize_answer(answer_is_match.group(1))

    # 5. Fallback to the last number in the string
    # Find all numbers (including decimals, negatives, and fractions like '1/9') in the output.
    numbers = re.findall(r"[-+]?\d+(?:/\d+)?|\d*\.\d+", model_output)
    if numbers:
        return normalize_answer(numbers[-1])

    return None

def is_correct(model_answer, golden_answer):
    """
    Compares the model's extracted answer with the golden answer.
    It handles single numbers, fractions, and comma-separated lists (order-insensitive).
    """
    # Extract the final answer from the model's detailed explanation
    extracted_answer = extract_final_answer(model_answer)
    
    # Process the golden answer to get the final numerical value
    golden_str = str(golden_answer)
    if "####" in golden_str:
        final_golden = golden_str.split("####")[-1].strip()
    else:
        final_golden = golden_str
        
    normalized_golden = normalize_answer(final_golden)

    if extracted_answer is None:
        print("-> Could not extract an answer from the model's output.")
        return False

    # Handle comma-separated lists (order-insensitive)
    if "," in extracted_answer and "," in normalized_golden:
        try:
            # Split, strip whitespace, convert to float, and sort
            extracted_list = sorted([float(x.strip()) for x in extracted_answer.split(',')])
            golden_list = sorted([float(x.strip()) for x in normalized_golden.split(',')])
            return extracted_list == golden_list
        except (ValueError, TypeError):
            # Fallback to string comparison if conversion fails
            return extracted_answer == normalized_golden

    # Handle single numbers and fractions
    try:
        # Evaluate fractions and compare as floating point numbers
        eval_extracted = float(eval(extracted_answer)) if "/" in extracted_answer else float(extracted_answer)
        eval_golden = float(eval(normalized_golden)) if "/" in normalized_golden else float(normalized_golden)
        return abs(eval_extracted - eval_golden) < 1e-6
    except (ValueError, TypeError, SyntaxError):
        # If conversion or evaluation fails, fall back to string comparison
        return extracted_answer == normalized_golden


def main():
    parser = argparse.ArgumentParser(description="Calculate final metrics for MoA-vllm evaluation.")
    parser.add_argument("--benchmark", "-b", required=True, help="Which benchmark we use to evaluate")
    parser.add_argument("--result_path", "-r", required=True, help="Path to the result file")
    args = parser.parse_args()
    benchmark = args.benchmark
    result_path = args.result_path
    
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Result file {result_path} does not exist.")
    print("Loading results from:", result_path)
    results = []
    with open(result_path, "r") as f:
        for line in f:
            results.append(json.loads(line.strip()))

    correct = 0
    wrong_items = []
    for entry in results:
        model_generated_answer = entry["answer"]
        golden_answer_key = entry["golden_answer"]
        if is_correct(model_generated_answer, golden_answer_key):
            correct += 1
        else:
            wrong_items.append(entry)            
    print(f"Total correct percent: {correct*100/len(results)}%")

    output_path = os.path.splitext(result_path)[0] + "_results.json"
    print("Saving results to:", output_path)
    with open(output_path, "w") as f:
        f.write(f"Total correct percent: {correct}/{len(results) * 100}%\n")
        json.dump(wrong_items, f, indent=4)

if __name__ == "__main__":
    main()

