from typing import List, Tuple, Set, Dict
from dataclasses import dataclass
from enum import Enum
import heapq

class StepType(Enum):
    MILESTONE = "milestone"    
    OPERATION = "operation"    
    CHECK = "check"
    STRATEGY = "strategy"
    HEURISTIC = "heuristic"
    BACKTRACK = "backtrack"

@dataclass
class SearchStep:
    step_type: StepType    
    nums: List[float]      
    operation: str        
    result: float         
    depth: int           
    reasoning: str
    cot: str
    heuristic_score: float = 0.0

@dataclass(order=True)
class SearchState:
    priority: float
    nums: List[float]
    expressions: List[str]
    depth: int
    path_reasoning: List[str] = None
    
    def __post_init__(self):
        if self.path_reasoning is None:
            self.path_reasoning = []
    
    def __hash__(self):
        return hash(tuple(sorted(self.nums)))

class Point24Solver:
    def __init__(self):
        self.TARGET = 24
        self.EPSILON = 1e-6
        self.steps = []
        self.found_solution = False
        self.solution = ""
        self.current_depth = 0
        self.operations_tried = 0
        self.best_intermediate = float('inf')
        
    def _format_num(self, num: float) -> str:
        if abs(num - round(num)) < self.EPSILON:
            return str(int(round(num)))
        return f"{num:.1f}"
        
    def _format_nums_list(self, nums: List[float]) -> str:
        return ', '.join(self._format_num(x) for x in nums)
        
    def _heuristic(self, nums: List[float]) -> float:

        if not nums:
            return float('inf')
            
        min_distance = min(abs(n - self.TARGET) for n in nums)
        
        operations_needed = len(nums) - 1
        
        promising_pairs = 0
        for i, n1 in enumerate(nums):
            for n2 in nums[i+1:]:
                results = [n1 + n2, n1 * n2]
                if n2 != 0:
                    results.append(n1 / n2)
                if n1 != 0:
                    results.append(n2 / n1)
                results.append(abs(n1 - n2))
                
                if any(abs(r - self.TARGET) < min_distance for r in results):
                    promising_pairs += 1
                    
        return min_distance + operations_needed * 2 - promising_pairs * 0.5
        
    def _generate_cot(self, current_nums: List[float], operation: str, result: float, 
                    heuristic_score: float, depth: int) -> str:
        # Simplified chain of thought
        # Only retain essential information
        parts = []
        parts.append(f"Current: [{self._format_nums_list(current_nums)}]")

        if operation != "Start":
            diff = abs(result - self.TARGET)
            if diff < self.best_intermediate:
                self.best_intermediate = diff
                parts.append("Better distance!")
            parts.append(f"Distance: {diff:.1f}")

        remaining_ops = len(current_nums) - 1
        parts.append(f"Operations left: {remaining_ops}")

        # Minimal suggestions
        if any(n >= self.TARGET for n in current_nums):
            parts.append("Consider dividing or subtracting.")
        else:
            parts.append("Consider adding or multiplying.")

        return " | ".join(parts)

    def _evaluate_path_quality(self, heuristic_score: float) -> str:
        if heuristic_score < 2:
            return "Very close!"
        elif heuristic_score < 5:
            return "Good path"
        elif heuristic_score < 10:
            return "Keep trying"
        else:
            return "Try another way"
    
    def _add_pattern_recognition(self, nums: List[float], cot_parts: List[str]):
        """Recognize special patterns in numbers"""
        products = []
        for i, n1 in enumerate(nums):
            for j, n2 in enumerate(nums[i+1:], i+1):
                if abs(n1 * n2 - self.TARGET) < 5:
                    products.append((n1, n2))
        if products:
            pairs = ", ".join([f"{self._format_num(x)}×{self._format_num(y)}" for x, y in products])
            cot_parts.append(f"Found potential multiplication pairs: {pairs}")
        
        for n1 in nums:
            for n2 in nums:
                if n2 != 0 and abs(n1 / n2 - self.TARGET) < 1:
                    cot_parts.append(f"Found potential division pair: {self._format_num(n1)}÷{self._format_num(n2)}")
        
        # Addition patterns
        sums = []
        for i, n1 in enumerate(nums):
            for j, n2 in enumerate(nums[i+1:], i+1):
                if abs(n1 + n2 - self.TARGET) < 5:
                    sums.append((n1, n2))
        if sums:
            pairs = ", ".join([f"{self._format_num(x)}+{self._format_num(y)}" for x, y in sums])
            cot_parts.append(f"Found potential addition pairs: {pairs}")

    def solve(self, nums: List[int]) -> Tuple[bool, str, List[SearchStep]]:
        self.steps.clear()
        self.found_solution = False
        self.solution = ""
        self.current_depth = 0
        self.operations_tried = 0
        self.best_intermediate = float('inf')
        
        initial_h_score = self._heuristic(nums)
        initial_cot = self._generate_cot(
            nums, "Start", 0, initial_h_score, 0
        )
        
        self.add_step(
            StepType.MILESTONE,
            nums,
            "Start",
            0,
            f"Start A* search, initial numbers: [{self._format_nums_list(nums)}]",
            initial_cot,
            initial_h_score
        )

        initial_state = SearchState(
            priority=initial_h_score,
            nums=nums,
            expressions=[str(x) for x in nums],
            depth=0
        )
        
        pq = [initial_state]
        visited = set()
        
        while pq:
            current = heapq.heappop(pq)
            state_hash = hash(tuple(sorted(current.nums)))
            
            if state_hash in visited:
                continue
                
            visited.add(state_hash)
            self.current_depth = current.depth
            
            if len(current.nums) == 1:
                if abs(current.nums[0] - self.TARGET) < self.EPSILON:
                    self.found_solution = True
                    self.solution = current.expressions[0]
                    
                    final_cot = self._generate_cot(
                        current.nums,
                        "Final Check",
                        current.nums[0],
                        0.0,
                        current.depth
                    )
                    
                    self.add_step(
                        StepType.CHECK,
                        current.nums,
                        "Success",
                        current.nums[0],
                        f"Solution Found: {self._format_num(current.nums[0])} = 24",
                        final_cot,
                        0.0
                    )
                    break
                continue
            
            for i in range(len(current.nums)):
                for j in range(i + 1, len(current.nums)):
                    num1, num2 = current.nums[i], current.nums[j]
                    expr1, expr2 = current.expressions[i], current.expressions[j]

                    remaining_nums = [current.nums[k] for k in range(len(current.nums)) if k != i and k != j]
                    remaining_expr = [current.expressions[k] for k in range(len(current.nums)) if k != i and k != j]
                    
                    # 尝试基本运算
                    operations = [
                        (lambda x, y: x + y, "+"),
                        (lambda x, y: x * y, "*"),
                        (lambda x, y: x - y, "-"),
                    ]
                    
                    for op_func, op_symbol in operations:
                        self.operations_tried += 1
                        result = op_func(num1, num2)
                        new_expr = f"({expr1} {op_symbol} {expr2})"
                        
                        new_nums = remaining_nums + [result]
                        new_expressions = remaining_expr + [new_expr]
                        
                        new_h_score = self._heuristic(new_nums)
                        operation_str = f"{self._format_num(num1)} {op_symbol} {self._format_num(num2)}"
                        
                        cot = self._generate_cot(
                            new_nums,
                            operation_str,
                            result,
                            new_h_score,
                            current.depth + 1
                        )
                        
                        if abs(result - self.TARGET) < abs(num1 - self.TARGET):
                            self.add_step(
                                StepType.OPERATION,
                                new_nums,
                                operation_str,
                                result,
                                f"Got {self._format_num(result)}, Distance to target: {abs(result - self.TARGET):.1f}",
                                cot,
                                new_h_score
                            )
                        
                        new_state = SearchState(
                            priority=new_h_score,
                            nums=new_nums,
                            expressions=new_expressions,
                            depth=current.depth + 1
                        )
                        heapq.heappush(pq, new_state)
                    
                    for x, y, expr_x, expr_y in [(num1, num2, expr1, expr2), 
                                               (num2, num1, expr2, expr1)]:
                        if abs(y) > self.EPSILON:
                            self.operations_tried += 1
                            result = x / y
                            new_expr = f"({expr_x} / {expr_y})"
                            
                            new_nums = remaining_nums + [result]
                            new_expressions = remaining_expr + [new_expr]
                            
                            new_h_score = self._heuristic(new_nums)
                            operation_str = f"{self._format_num(x)} / {self._format_num(y)}"
                            
                            cot = self._generate_cot(
                                new_nums,
                                operation_str,
                                result,
                                new_h_score,
                                current.depth + 1
                            )
                            
                            if abs(result - self.TARGET) < abs(x - self.TARGET):
                                self.add_step(
                                    StepType.OPERATION,
                                    new_nums,
                                    operation_str,
                                    result,
                                    f"Got {self._format_num(result)}, Distance to target: {abs(result - self.TARGET):.1f}",
                                    cot,
                                    new_h_score
                                )
                            
                            new_state = SearchState(
                                priority=new_h_score,
                                nums=new_nums,
                                expressions=new_expressions,
                                depth=current.depth + 1
                            )
                            heapq.heappush(pq, new_state)
        
        return self.found_solution, self.solution, self.steps
        
    def add_step(self, step_type: StepType, nums: List[float], operation: str, 
                result: float, reasoning: str, cot: str, heuristic_score: float = 0.0):
        self.steps.append(SearchStep(
            step_type=step_type,
            nums=nums.copy(),
            operation=operation,
            result=result,
            depth=self.current_depth,
            reasoning=reasoning,
            cot=cot,
            heuristic_score=heuristic_score
        ))

    def format_solution_with_cot(self, nums: List[int], found: bool, solution: str, steps: List[SearchStep]) -> str:
        """Format solution with complete CoT reasoning process"""
        result = []
        
        result.append(f"Problem: Calculate 24 using numbers {nums}")
        
        for i, step in enumerate(steps):
            result.append(f"\nOperation: {step.operation}")
            result.append(f"Reasoning: {step.reasoning}")
            result.append(step.cot)
        
        if found:
            result.append(f"Solution: ```{solution}```")
        else:
            result.append("Solution: ```None```")
            
        return '\n'.join(result)

def demo_solution_with_cot(nums: List[int]):
    solver = Point24Solver()
    found, solution, steps = solver.solve(nums)
    return solver.format_solution_with_cot(nums, found, solution, steps)

if __name__ == "__main__":
    nums = [2, 3, 8, 12]
    print(len(demo_solution_with_cot(nums).split()))
