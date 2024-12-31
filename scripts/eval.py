import argparse
import json
import re
from collections import Counter

def extract_answer(example: dict):
    response = example["response"]
    code_block = re.search(r"```(.*)```", response)
    if code_block:
        return code_block.group(1)
    return 'None'

def process_data(file_path: str):
    examples = []
    with open(file_path, "r") as f:
        for line in f:
            example = json.loads(line)
            answer = extract_answer(example)
            examples.append({"puzzle": example["puzzle"], "answer": answer})
    return examples

def normalize_answer(expr: str):
    expr = expr.split('=')[0]
    expr = expr.replace(' ', '')
    
    expr = expr.replace('\\times', '*')
    expr = expr.replace('×', '*')
    expr = expr.replace('x', '*')
    
    expr = expr.replace('\\div', '/')
    expr = expr.replace('÷', '/')
    
    return expr

def evaluate_expression(expr: str, puzzle: str) -> bool:
    try:
        if not verify_numbers(puzzle, expr):
            return False

        result = eval(expr)
        return abs(result - 24) < 1e-6
    except Exception as e:
        return False

def extract_numbers(expr: str) -> list:
    return [int(num) for num in re.findall(r'\d+', expr)]

def verify_numbers(puzzle: str, expr: str) -> bool:
    puzzle_nums = [int(x.strip()) for x in puzzle.split(',')]
    expr_nums = extract_numbers(expr)

    return Counter(puzzle_nums) == Counter(expr_nums)

def evaluate(file_path: str):
    examples = process_data(file_path)
    correct = 0
    total = 0
    
    for example in examples:
        answer = normalize_answer(example["answer"])
        is_correct = evaluate_expression(answer, example["puzzle"])
        example["correct"] = is_correct
        example["normalized_answer"] = answer

        total += 1
        if is_correct:
            correct += 1

    print(f"Total: {total}, Correct: {correct}")
    return examples

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    args = parser.parse_args()
    examples = evaluate(args.file)
    # save examples to csv
    import pandas as pd
    df = pd.DataFrame(examples)
    df.to_csv("eval_result.csv", index=False)
