import os
from openai import OpenAI
import argparse
import json
import re
from collections import Counter
import pandas as pd

from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

SYSTEM_PROMPT = '''
You are a 24 Game assistant. Given 4 numbers (range 1-13), you need to use basic arithmetic operations (+, -, *, /) to combine these numbers to get 24.

**Rules:**
1. You MUST USE all 4 given numbers EXACTLY ONCE, even if numbers are REPEATED.
   - For example, if the input is 4, 4, 5, 6, you must use 4, 4, 5, 6 exactly once.
   - If the equation is equal to 24 but does not use all 4 numbers exactly once, it is invalid and SHOULD NOT be considered as a solution.
   - You CANNOT use other numbers except for the 4 given numbers.
2. Only these operators are allowed: addition(+), subtraction(-), multiplication(*), division(/)
3. Parentheses can be used to change operation precedence
5. The result must equal exactly 24, or be approximately 24 within calculation error (e.g., 23.9999...)

**Output Requirements:**

Please reason step by step, show your reasoning process and put your final answer within \\boxed{}. Avoid using LaTeX expression in the final answer and use (+, -, *, /) instead.
'''

def extract_answer(example: str):
    # extract from \boxed{}
    answer = re.search(r"\\boxed{(.*)}", example)
    if answer:
        return answer.group(1)
    else:
        return None

def normalize_answer(expr: str):
    expr = expr.split('=')[0]
    expr = expr.replace(' ', '')
    
    expr = expr.replace('\\times', '*')
    expr = expr.replace('×', '*')
    expr = expr.replace('x', '*')
    
    expr = expr.replace('\\div', '/')
    expr = expr.replace('÷', '/')

    expr = expr.strip()
    expr = expr.replace('\\', '')
    expr = expr.replace('\n', '')
    expr = expr.replace(' ', '')
    
    return expr

def evaluate_expression(expr: str, puzzle: str) -> bool:
    try:
        if not verify_numbers(puzzle, expr):
            return False

        if not all(c in '0123456789+-*/(). ' for c in expr):
            return False

        result = eval(expr)
        return abs(result - 24) < 1e-6
    except Exception as e:
        print(e)
        return False

def extract_numbers(expr: str) -> list:
    return [int(num) for num in re.findall(r'\d+', expr)]

def verify_numbers(puzzle: str, expr: str) -> bool:
    puzzle_nums = [int(x.strip()) for x in puzzle.split(' ')]
    expr_nums = extract_numbers(expr)

    return Counter(puzzle_nums) == Counter(expr_nums)

def generate_response(instruction: str):
    
    # add retry
    max_retry = 3
    retry = 0

    while retry < max_retry:
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{
                "role": "system",
                "content": SYSTEM_PROMPT
                }, {
                "role": "user",
                "content": instruction
            }],
                temperature=0.1
            )
            if response is not None:
                break
        except Exception as e:
            retry += 1
            continue

    if retry == max_retry:
        return None
    else:
        return response.choices[0].message.content


output_data_lock = Lock()
output_failed_lock = Lock()

output_data = []
output_failed_data = []

def process_puzzle(puzzle: str):

    max_retry = 5
    retry = 0

    instruction = f"Solve the problem with the numbers {puzzle}"

    # if not correct, retry
    for _ in range(max_retry):
        try:
            response = generate_response(instruction)
            answer = extract_answer(response)
            if answer is not None:
                answer = normalize_answer(answer)
                is_correct = evaluate_expression(answer, puzzle)
            else:
                answer = None
                is_correct = False
        except Exception as e:
            print(e)
            continue

        if is_correct:
            with output_data_lock:
                output_data.append({
                    "instruction": instruction,
                    "response": response,
                    "puzzle": puzzle,
                    "answer": answer,
                    "correct": is_correct
                })
            break
        else:
            with output_failed_lock:
                output_failed_data.append({
                    "instruction": instruction,
                    "response": response,
                    "answer": answer,
                    "correct": is_correct
                })
 
    return answer

if __name__ == "__main__":
    
    test_data = pd.read_csv("../data/train_df.csv")
    puzzles = test_data["Puzzles"].tolist()

    with open("../data/deepseek/output_data.json", "r") as f:
        generated_puzzles = json.load(f)
        generated_puzzles = [item["puzzle"] for item in generated_puzzles]

    puzzles = [puzzle for puzzle in puzzles if puzzle not in generated_puzzles]

    SAVE_INTERVAL = 100
    processed_count = 0
    
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(process_puzzle, puzzle) for puzzle in puzzles]
        for future in tqdm(as_completed(futures), total=len(puzzles), desc="Processing puzzles"):
            pass