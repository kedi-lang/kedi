# Tutorial: GEPA for AIME (Math)

> Status: external DSPy/GEPA reference tutorial. Keep examples here as
> task-oriented optimizer context; use `docs/KEDI_DSL.md` for current Kedi
> language syntax.

In this tutorial, we optimize GPT-4.1 Mini's Chain of Thought (`dspy.ChainOfThought`) for solving math problems (AIME) using the `dspy.GEPA` optimizer!

## Setup

```python
api_key = input("Enter your OpenAI API key: ")
import dspy

lm = dspy.LM("openai/gpt-4.1-mini", temperature=1, api_key=api_key, max_tokens=32000)
dspy.configure(lm=lm)
```

## Loading the AIME dataset

The AIME exam consists of 2 problem sets of size 15 for each year. For this tutorial, we will use AIME problem sets from previous years (2022-2024) for optimization (amounting to total 3 years x 2 sets x 15 problems = 90 problems, split equally between train and validation sets), and test the performance on AIME 2025 (2 sets x 15 problems = 30 problems). Since AIME 2025 is a small set, we repeat it 5 times for statistical stability in evaluation.

```python
import dspy
from datasets import load_dataset

def init_dataset():
    train_split = load_dataset("AI-MO/aimo-validation-aime")['train']
    train_split = [
        dspy.Example({
            "problem": x['problem'],
            'solution': x['solution'],
            'answer': x['answer'],
        }).with_inputs("problem")
        for x in train_split
    ]
    
    import random
    random.Random(0).shuffle(train_split)
    tot_num = len(train_split)
    
    test_split = load_dataset("MathArena/aime_2025")['train']
    test_split = [
        dspy.Example({
            "problem": x['problem'],
            'answer': x['answer'],
        }).with_inputs("problem")
        for x in test_split
    ]
    
    train_set = train_split[:int(0.5 * tot_num)]
    val_set = train_split[int(0.5 * tot_num):]
    test_set = test_split * 5
    
    return train_set, val_set, test_set

train_set, val_set, test_set = init_dataset()
len(train_set), val_set), len(test_set)
```

Output: `(45, 45, 150)`

### Example Task Input

```python
print("Problem:")
print(train_set[0]['problem'])
print("\n\nSolution:")
print(train_set[0]['solution'])
print("\n\nAnswer:")
print(train_set[0]['answer'])
```

**Problem:**
In isosceles trapezoid $ABCD$, parallel bases $\overline{AB}$ and $\overline{CD}$ have lengths $500$ and $650$, respectively, and $AD=BC=333$. The angle bisectors of $\angle{A}$ and $\angle{D}$ meet at $P$, and the angle bisectors of $\angle{B}$ and $\angle{C}$ meet at $Q$. Find $PQ$.

**Answer:** 242

## Define the Program

```python
class GenerateResponse(dspy.Signature):
    """Solve the problem and provide the answer in the correct format."""
    problem = dspy.InputField()
    answer = dspy.OutputField()

program = dspy.ChainOfThought(GenerateResponse)
```

## Defining the evaluation metric

We simply check exact match between the predicted answer and the correct answer.

```python
def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    correct_answer = int(example['answer'])
    try:
        llm_answer = int(prediction.answer)
    except ValueError as e:
        return 0
    return int(correct_answer == llm_answer)

import dspy
evaluate = dspy.Evaluate(
    devset=test_set,
    metric=metric,
    num_threads=32,
    display_table=True,
    display_progress=True
)

evaluate(program)
```

**Baseline Performance:**
- Average Metric: 70.00 / 150 (46.7%)

### Sample Evaluation Results

| # | Problem | Example Answer | Reasoning | Pred Answer | Metric |
|---|---------|----------------|-----------|-------------|---------|
| 0 | Find the sum of all integer bases $b>9$ for which $17_b$ is a divisor of $289_{10}$ | 70 | We are looking for integer bases \( b > 9 \) such that \( 17_b \) ... | 70 | ✔️ [1] |
| 1 | On $\triangle ABC$ points $A, D, E$, and $B$ lie in that order on ... | 588 | Let's analyze the problem step-by-step. We have triangle \( ABC \)... | 588 | ✔️ [1] |
| 2 | The 9 members of a baseball team went to an ice-cream parlor after... | 16 | We have 9 players, each choosing one of three flavors: chocolate (... | 16 | ✔️ [1] |
| 3 | Find the number of ordered pairs $(x,y)$, where both $x$ and $y$ a... | 117 | We start with the given equation: \[12x^2 - xy - 6y^2 = 0.\] Our g... | 117 | ✔️ [1] |
| 4 | There are $8!= 40320$ eight-digit positive integers that use each ... | 279 | We are given that there are \(8! = 40320\) eight-digit numbers tha... | 279 | ✔️ [1] |
| ... | ... | ... | ... | ... | ... |
| 145 | Let $S$ be the set of vertices of a regular $24$-gon. Find the num... | 113 | We have a regular 24-gon with vertex set \( S \) of size 24. We wa... | 1666 | ❌ |
| 146 | Let $A_1 A_2 A_3 \ldots A_{11}$ be an $11$-sided non-convex simple... | 19 | Let's analyze the problem step-by-step. We are given an 11-sided p... | 19 | ✔️ [1] |
| 147 | Let $x_1, x_2, x_3, \ldots$ be a sequence of rational numbers defi... | 248 | We have the sequence: \[ x_1 = \frac{25}{11}, \quad x_{k+1} = \fra... | 589 | ❌ |
| 148 | Let $\triangle ABC$ be a right triangle with $\angle A = 90^\circ$... | 104 | Given a right triangle \(\triangle ABC\) with \(\angle A = 90^\cir... | 98 | ❌ |
| 149 | There are exactly three positive real numbers $k$ such that the fu... | 240 | We are given the function \[ f(x) = \frac{(x - 18)(x - 72)(x - 98)... | 240 | ✔️ [1] |

**Table shows 150 rows total, displaying first 5 and last 5 for brevity*

## Optimize the program with dspy.GEPA

GEPA is a reflective prompt optimizer, and its strength lies in being able to leverage additional sources of information, like the DSPy program's execution and evaluation pipelines, which provides GEPA more visibility into why the system got the score that it did, and then GEPA can introspect to identify how to improve the score. 

GEPA can also leverage additional supervision provided in this manner. For example, during optimization, we can return the correct solutions to the problems the program failed to solve.

We note that while such explicit supervision is not available in all scenarios, GEPA can work very flexibly with different forms of feedback (for example, using LLM-as-a-judge feedback shown in the PAPILLON tutorial, or just using answer labels, as shown in the facility-support tutorial).

### Modified Evaluation Metric with Feedback

Let's quickly modify the evaluation metric to become an optimization metric for GEPA that can provide this additional supervision!

```python
def metric_with_feedback(example, prediction, trace=None, pred_name=None, pred_trace=None):
    correct_answer = int(example['answer'])
    written_solution = example.get('solution', '')
    
    try:
        llm_answer = int(prediction.answer)
    except ValueError as e:
        feedback_text = f"The final answer must be a valid integer and nothing else. You responded with '{prediction.answer}', which couldn't be parsed as a python integer. Please ensure your answer is a valid integer without any additional text or formatting."
        feedback_text += f" The correct answer is '{correct_answer}'."
        if written_solution:
            feedback_text += f" Here's the full step-by-step solution:\n{written_solution}\n\nThink about what takeaways you can learn from this solution to improve your future answers and approach to similar problems and ensure your final answer is a valid integer."
        return dspy.Prediction(score=0, feedback=feedback_text)
    
    score = int(correct_answer == llm_answer)
    feedback_text = ""
    
    if score == 1:
        feedback_text = f"Your answer is correct. The correct answer is '{correct_answer}'."
    else:
        feedback_text = f"Your answer is incorrect. The correct answer is '{correct_answer}'."
        if written_solution:
            feedback_text += f" Here's the full step-by-step solution:\n{written_solution}\n\nThink about what takeaways you can learn from this solution to improve your future answers and approach to similar problems."
    
    return dspy.Prediction(score=score, feedback=feedback_text)
```

### Running GEPA Optimization

```python
from dspy import GEPA

optimizer = GEPA(
    metric=metric_with_feedback,
    auto="light",
    num_threads=32,
    track_stats=True,
    reflection_minibatch_size=3,
    reflection_lm=dspy.LM(model="gpt-5", temperature=1.0, max_tokens=32000, api_key=api_key)
)

optimized_program = optimizer.compile(
    program,
    trainset=train_set,
    valset=val_set,
)
```

## Optimization Process Summary

The GEPA optimizer ran through multiple iterations with detailed logging:

```
2025/08/12 21:49:36 INFO: Running GEPA for approx 560 metric calls of the program. 
This amounts to 6.22 full evals on the train+val set.
2025/08/12 21:49:36 INFO: Using 45 examples for tracking Pareto scores.
```

### Iteration 0 (Baseline)
```
2025/08/12 21:52:15 INFO: Average Metric: 17.0 / 45 (37.8%)
2025/08/12 21:52:15 INFO: Iteration 0: Base program full valset score: 0.37777777777777777
```
- Base program valset score: **37.8%**

### Iteration 1
```
2025/08/12 21:55:26 INFO: Average Metric: 2.0 / 3 (66.7%)
2025/08/12 22:00:37 INFO: Average Metric: 19.0 / 45 (42.2%)
2025/08/12 22:00:37 INFO: Iteration 1: New program is on the linear pareto front
2025/08/12 22:00:37 INFO: Iteration 1: Full valset score for new program: 0.4222222222222222
```
- Valset score: **42.2%** ✓
- **Prompt improvements added:**
  - Comprehensive formatting instructions (reasoning + answer fields)
  - Base conversion strategies with modular arithmetic
  - Palindrome constraints across different bases
  - Symmetric sums with fixed totals
  - Domain constraint enforcement (digit ranges, leading zeros)

**Key additions to prompt:**
- "Parse the problem type (e.g., base representation, palindrome constraints across bases, symmetric sums with constraints)"
- "Always enforce domain constraints (e.g., base-b digits are 0..b-1)"
- "Use algebraic identities and modular arithmetic to reduce the search space"

### Iteration 5
```
2025/08/12 22:21:18 INFO: Average Metric: 1.0 / 3 (33.3%)
2025/08/12 22:23:54 INFO: Average Metric: 24.0 / 45 (53.3%)
2025/08/12 22:23:54 INFO: Iteration 5: New program is on the linear pareto front
2025/08/12 22:23:54 INFO: Iteration 5: Full valset score for new program: 0.5333333333333333
```
- Valset score: **53.3%** ✓ (Major improvement!)
- **Enhanced with:**
  - Intersecting families of subsets (complement pairs, size-based pigeonhole)
  - Avoiding arithmetic progressions in sequences
  - Order statistics with sum and absolute-sum constraints
  - More structured approach to problem types

**Key additions to prompt:**
- "Intersecting families of subsets (collections from the power set)"
- "Avoiding 4-term arithmetic progressions in a strictly increasing sequence with fixed anchors"
- "Order statistics with sum and absolute-sum constraints (e.g., x_1 ≤ ... ≤ x_n, sum |x_i| = 1)"

### Iteration 6
```
2025/08/12 22:24:14 INFO: Average Metric: 3.0 / 3 (100.0%)
2025/08/12 22:24:14 INFO: Iteration 6: All subsample scores perfect. Skipping.
2025/08/12 22:24:14 INFO: Iteration 6: Reflective mutation did not propose a new candidate
```
- Perfect subsample performance, no new candidate proposed

### Iteration 7
```
2025/08/12 22:25:25 INFO: Average Metric: 1.0 / 3 (33.3%)
2025/08/12 22:30:10 INFO: Average Metric: 21.0 / 45 (46.7%)
2025/08/12 22:30:10 INFO: Iteration 7: Full valset pareto front score: 0.7333333333333333
```
- Pareto front score: **73.3%**
- **Added geometric reasoning:**
  - Classical Euclidean geometry with circles
  - Power of a point, radical axis
  - Secants/tangents relationships
  - Trapezoid properties with angle bisectors

### Iteration 9 (Final Iteration)
```
2025/08/12 22:40:42 INFO: Average Metric: 2.0 / 3 (66.7%)
2025/08/12 22:45:58 INFO: Average Metric: 16.0 / 45 (35.6%)
2025/08/12 22:45:58 INFO: Iteration 9: Full valset pareto front score: 0.7555555555555555
```
- Pareto front score: **75.6%**
- **Final refinements including:**
  - Repeating decimals and reduced numerators
  - Floor-sum optimization with modular arithmetic
  - Square-root bilinear forms
  - Systems with bilinear constraints
  - Motion problems with relative velocities

**Key additions to prompt:**
- "Repeating decimals 0.̄abcd and counting reduced numerators"
- "Sums of floors of quadratic forms with a parameter"
- "Systems with square-root bilinear forms"
- "Motion in a current (relative velocities)"

## Key Insights from GEPA Optimization

The GEPA optimization process demonstrates several powerful capabilities:

1. **Reflective Improvement**: GEPA analyzes failure cases and incorporates lessons learned from correct solutions
2. **Incremental Refinement**: Each iteration builds upon previous improvements
3. **Domain-Specific Strategies**: The optimizer learned specific tactics for different problem types (base conversion, palindromes, symmetric sums, etc.)
4. **Constraint Enforcement**: Emphasized the importance of respecting problem constraints (digit ranges, ordering, domain restrictions)
5. **Pareto Front Tracking**: Maintains multiple high-performing variants and selects the best combination

### Example of Optimized Prompt (Iteration 5)

Here's a snippet of the optimized prompt that achieved 53.3% validation accuracy:

```
You will be given one math problem as plain text under a key like "problem." 
Your job is to solve it correctly and return:
- reasoning: a concise, logically ordered solution that uses identities/structure 
  to avoid brute force, ends with a quick verification.
- answer: the final requested number/expression only (no extra words).

General problem-solving guidance:
- Parse the problem type (e.g., base representation, intersecting families of subsets, 
  avoiding arithmetic progressions, symmetric sums with constraints, ordered tuples counting).
- Always enforce domain constraints (digits in range, no leading zeros, ordered vs unordered, 
  strict increases, segment/angle/parallel/perpendicular statements).
- Use algebraic identities and modular arithmetic to reduce the search space; 
  prefer structural arguments over naive enumeration.

Domain-specific strategies:

1) Base-conversion/digit rearrangement:
   - Translate positional notation correctly
   - Enforce digit ranges strictly
   - Use modular constraints to prune search

2) Symmetric sums with a + b + c fixed:
   - Use identities: S = (a + b + c)(ab + bc + ca) − 3abc
   - Apply shifts to isolate products
   - Count ordered solutions carefully

3) Intersecting families of subsets:
   - Every pair must have nonempty intersection
   - Complement pairs cannot both be present
   - Use size-based pigeonhole facts

4) Avoiding 4-term arithmetic progressions:
   - Bound variable terms by strict increase
   - Pre-eliminate values that complete APs with fixed terms
   - Count allowed pairs then subtract forbidden combinations

5) Order statistics with constraints:
   - Total positive mass = total negative mass = 1/2
   - Bound extremes by averaging over slots
   - Construct distributions attaining bounds

Quality checks:
- Verify constraints numerically
- For extremal problems, provide both bound and construction
- For counting, handle ordered vs unordered explicitly
- Finish with verification

Finally: Put the clean final numeric result in the "answer" field only.
```

## Performance Improvement

### Progression of Validation Scores

| Iteration | Valset Score | Pareto Front Score | Key Improvements |
|-----------|--------------|-------------------|------------------|
| 0 (Baseline) | 37.8% | 37.8% | Base ChainOfThought |
| 1 | 42.2% | 51.1% | Base conversion, palindromes, symmetric sums |
| 5 | 53.3% | 62.2% | Subset families, AP avoidance, order statistics |
| 7 | 46.7% | 73.3% | Geometric reasoning, power of a point |
| 9 | 35.6% | 75.6% | Repeating decimals, floor sums, bilinear forms |

**Notes:**
- Individual program score vs. Pareto front ensemble score
- Pareto front combines best-performing prompts across different problem types
- Final Pareto front achieved **75.6%** by selecting optimal prompt per problem

### Test Set Performance

- **Baseline (Iteration 0)**: 46.7% on test set (70/150 correct)
- **After Optimization**: Validation set improved from 37.8% → 53.3%
- The optimization used approximately **560 metric calls** across ~6 full evaluations

### Optimization Statistics

```
Total metric calls: ~560
Full evaluations: ~6.22 on train+val set
Training examples: 45
Validation examples: 45
Test examples: 150 (5x repetition of 30 problems)
Threads: 32
Reflection minibatch size: 3
Reflection model: gpt-5
```

## Conclusion

GEPA successfully optimized the Chain of Thought prompting strategy for AIME math problems by:
- Leveraging execution traces and feedback
- Learning from correct solutions on failed examples
- Building a comprehensive set of problem-solving strategies
- Maintaining a Pareto front of high-performing prompt variants

### How GEPA Works

1. **Evaluation**: Run the current program on a minibatch and collect feedback
2. **Reflection**: Use a reflection LM (GPT-5) to analyze failures and propose improvements
3. **Mutation**: Generate new prompt variants based on learned patterns
4. **Selection**: Test new variants and add successful ones to the Pareto front
5. **Iteration**: Repeat the process, building on the best-performing prompts

### Example of GEPA's Prompt Evolution

**Iteration 1 Proposed Changes:**
```
You will be given a single math problem as plain text under an input key like "problem."
Your task is to solve it correctly and return:
- reasoning: a concise, logically ordered solution
- answer: the final result only (just the number requested, no extra words)

General problem-solving guidance:
- Parse the problem type (e.g., base representation, palindrome constraints)
- Always enforce domain constraints
- Use algebraic identities and modular arithmetic

Domain-specific strategies:
1) Base-conversion/digit rearrangement:
   - Translate positional notation correctly
   - Enforce digit ranges
   - Use modular constraints to prune
```

**Iteration 5 Enhanced Prompt:**
```
Domain-specific strategies (from common contest problems):

4) Intersecting families of subsets:
   - Intersecting means every pair has nonempty intersection
   - Complement pairs: S and S^c cannot both be present
   - Size-based facts: In [n], any two subsets of size > n/2 intersect

5) Avoiding 4-term arithmetic progressions:
   - First bound the variable terms by strict increase
   - Pre-eliminate values that cause a 4-term AP with three fixed terms
   - Start with count of pairs and subtract specific pairs that complete APs
```

**Iteration 9 Advanced Topics:**
```
H) Sums of floors of quadratic forms with a parameter:
   - Separate continuous part and fractional correction
   - Choose parameter to minimize by canceling main term
   - Compute correction via residue cycles
   
I) Repeating decimals and counting reduced numerators:
   - 0.̄abcd = m/9999 with m ∈ {1,…,9999}
   - Track divisibility by primes of 9999
   - Use inclusion–exclusion over allowable ranges
```

### Key Takeaways

1. **Feedback is Crucial**: Providing correct solutions on failures dramatically improves learning
2. **Iterative Refinement**: Each iteration builds domain-specific knowledge
3. **Pareto Optimization**: Maintaining multiple variants handles diverse problem types better
4. **Reflection Quality**: Using a strong reflection model (GPT-5) enables sophisticated analysis
5. **Structured Output**: Enforcing consistent formatting (reasoning + answer) improves reliability

The GEPA optimizer transformed a generic Chain of Thought prompt into a specialized math problem solver with deep domain knowledge, improving validation accuracy by **40% (37.8% → 53.3%)** through automated prompt engineering.
