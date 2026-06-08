# Tutorial: GEPA for Structured Information Extraction for Enterprise Tasks

> Status: external DSPy/GEPA reference tutorial. This file currently mirrors
> `docs/gepa_papillon.md`; keep both until the docs surface is reviewed.

In this tutorial, we'll explore a three-part task for structured information extraction and classification using the [Facility Support Analyzer](https://github.com/meta-llama/llama-prompt-ops/tree/main/use-cases/facility-support-analyzer) dataset released by Meta. Given an email or message sent in an enterprise setting related to facility maintenance or support requests, the goal is to extract its **urgency**, assess the **sentiment**, and identify all relevant **service request categories**.

We will build a simple DSPy program, and then use the `dspy.GEPA` optimizer to optimize it for the task.

## Setup

```python
api_key = input("Enter your OpenAI API key: ")
import dspy

lm = dspy.LM("openai/gpt-4.1-nano", temperature=1, api_key=api_key)
dspy.configure(lm=lm)
```

## Loading the Dataset

```python
import requests
import dspy
import json
import random

def init_dataset():
    # Load from the url
    url = "https://raw.githubusercontent.com/meta-llama/llama-prompt-ops/refs/heads/main/use-cases/facility-support-analyzer/dataset.json"
    dataset = json.loads(requests.get(url).text)
    
    dspy_dataset = [
        dspy.Example({
            "message": d['fields']['input'],
            "answer": d['answer'],
        }).with_inputs("message")
        for d in dataset
    ]
    
    random.Random(0).shuffle(dspy_dataset)
    
    train_set = dspy_dataset[:int(len(dspy_dataset) * 0.33)]
    val_set = dspy_dataset[int(len(dspy_dataset) * 0.33):int(len(dspy_dataset) * 0.66)]
    test_set = dspy_dataset[int(len(dspy_dataset) * 0.66):]
    
    return train_set, val_set, test_set

train_set, val_set, test_set = init_dataset()
len(train_set), len(val_set), len(test_set)
```

**Output:** `(66, 66, 68)`

### Example Task Input

```python
print("Input Message:")
print(train_set[0]['message'])
print("\n\nGold Answer:")
for k, v in json.loads(train_set[0]['answer']).items():
    print(f"{k}: {v}")
```

**Input Message:**
```
Subject: Adjusting Bi-Weekly Cleaning Schedule for My Office 

Dear ProCare Facility Solutions Support Team,

I hope this message finds you well. My name is Dr. Alex Turner, and I have been utilizing your 
services for my office space for the past year. I must say, your team's dedication to maintaining 
a pristine environment has been commendable and greatly appreciated.

I am reaching out to discuss the scheduling of our regular cleaning services. While I find the 
logistical challenges of coordinating these services intellectually stimulating, I believe we 
could optimize the current schedule to better suit the needs of my team and our workflow. 
Specifically, I would like to explore the possibility of adjusting our cleaning schedule to a 
bi-weekly arrangement, ideally on Tuesdays and Fridays, to ensure our workspace remains 
consistently clean without disrupting our research activities.

Previously, I have attempted to adjust the schedule through the online portal, but I encountered 
some difficulties in finalizing the changes. I would appreciate your assistance in making these 
adjustments or guiding me through the process if there is a more efficient way to do so.

Thank you for your attention to this matter. I look forward to your response and continued 
excellent service.

Best regards,
Dr. Alex Turner
Cryptography Researcher
```

**Gold Answer:**
```
categories: {'routine_maintenance_requests': False, 'customer_feedback_and_complaints': False, 
'training_and_support_requests': False, 'quality_and_safety_concerns': False, 
'sustainability_and_environmental_practices': False, 'cleaning_services_scheduling': True, 
'specialized_cleaning_services': False, 'emergency_repair_services': False, 
'facility_management_issues': False, 'general_inquiries': False}
sentiment: neutral
urgency: low
```

## Defining a DSPy Program to Solve the Task

The program is a 3-module system, each of which handles the urgency, sentiment and categories classification respectively.

```python
from typing import List, Literal

class FacilitySupportAnalyzerUrgency(dspy.Signature):
    """
    Read the provided message and determine the urgency.
    """
    message: str = dspy.InputField()
    urgency: Literal['low', 'medium', 'high'] = dspy.OutputField()

class FacilitySupportAnalyzerSentiment(dspy.Signature):
    """
    Read the provided message and determine the sentiment.
    """
    message: str = dspy.InputField()
    sentiment: Literal['positive', 'neutral', 'negative'] = dspy.OutputField()

class FacilitySupportAnalyzerCategories(dspy.Signature):
    """
    Read the provided message and determine the set of categories applicable to the message.
    """
    message: str = dspy.InputField()
    categories: List[Literal["emergency_repair_services", "routine_maintenance_requests", 
                             "quality_and_safety_concerns", "specialized_cleaning_services", 
                             "general_inquiries", "sustainability_and_environmental_practices", 
                             "training_and_support_requests", "cleaning_services_scheduling", 
                             "customer_feedback_and_complaints", "facility_management_issues"]] = dspy.OutputField()

class FacilitySupportAnalyzerMM(dspy.Module):
    def __init__(self):
        self.urgency_module = dspy.ChainOfThought(FacilitySupportAnalyzerUrgency)
        self.sentiment_module = dspy.ChainOfThought(FacilitySupportAnalyzerSentiment)
        self.categories_module = dspy.ChainOfThought(FacilitySupportAnalyzerCategories)
    
    def __call__(self, message: str):
        urgency = self.urgency_module(message=message)
        sentiment = self.sentiment_module(message=message)
        categories = self.categories_module(message=message)
        
        return dspy.Prediction(
            urgency=urgency.urgency,
            sentiment=sentiment.sentiment,
            categories=categories.categories
        )

program = FacilitySupportAnalyzerMM()
```

## Define the Metric to Evaluate the Outputs

The metric evaluates the output of all three tasks and returns the aggregate score.

```python
def score_urgency(gold_urgency, pred_urgency):
    """Compute score for the urgency module."""
    score = 1.0 if gold_urgency == pred_urgency else 0.0
    return score

def score_sentiment(gold_sentiment, pred_sentiment):
    """Compute score for the sentiment module."""
    score = 1.0 if gold_sentiment == pred_sentiment else 0.0
    return score

def score_categories(gold_categories, pred_categories):
    """Compute score for the categories module."""
    correct = 0
    for k, v in gold_categories.items():
        if v and k in pred_categories:
            correct += 1
        elif not v and k not in pred_categories:
            correct += 1
    score = correct / len(gold_categories)
    return score

def metric(example, pred, trace=None, pred_name=None, pred_trace=None):
    """
    Computes a score based on agreement between prediction and gold standard 
    for categories, sentiment, and urgency. Returns the score (float).
    """
    # Parse gold standard from example
    gold = json.loads(example['answer'])
    
    # Compute scores for all modules
    score_urgency_val = score_urgency(gold['urgency'], pred.urgency)
    score_sentiment_val = score_sentiment(gold['sentiment'], pred.sentiment)
    score_categories_val = score_categories(gold['categories'], pred.categories)
    
    # Overall score: average of the three accuracies
    total = (score_urgency_val + score_sentiment_val + score_categories_val) / 3
    return total

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

## Baseline Performance

```
Average Metric: 51.30 / 68 (75.4%)
```

### Sample Evaluation Results

| # | Message | Answer | Urgency | Sentiment | Categories | Metric |
|---|---------|--------|---------|-----------|------------|--------|
| 0 | Hey ProCare Support Team, Hope you all are doing great!... | {...} | low | positive | [sustainability_and_environmental_practices] | ✔️ [1.000] |
| 1 | Hey ProCare Team, Hope you're all doing well!... | {...} | medium | positive | [routine_maintenance_requests, customer_feedback_and_complaints] | ✔️ [0.967] |
| 2 | Subject: Assistance Needed for HVAC Maintenance... | {...} | medium | neutral | [routine_maintenance_requests] | ✔️ [1.000] |
| 3 | Subject: A Green Inquiry from a Bill Maher Enthusiast... | {...} | low | positive | [sustainability_and_environmental_practices] | ✔️ [1.000] |
| 4 | Subject: Inquiry on Sustainability Practices... | {...} | medium | neutral | [sustainability_and_environmental_practices] | ✔️ [0.667] |

**68 rows total**

## Adding Feedback for GEPA Optimization

GEPA is a reflective prompt optimizer. Its strength lies in its ability to examine textual feedback from the DSPy program's execution and evaluation pipelines. This gives GEPA greater insight into why the system achieved a particular score, enabling it to introspect and determine ways to enhance performance.

GEPA supports providing feedback at the individual predictor level. Let's modify our evaluation metric to provide text feedback!

```python
import json
import dspy

def feedback_urgency(gold_urgency, pred_urgency):
    """Generate feedback for the urgency module."""
    score = 1.0 if gold_urgency == pred_urgency else 0.0
    
    if gold_urgency == pred_urgency:
        feedback = f"You correctly classified the urgency of the message as `{gold_urgency}`. This message is indeed of `{gold_urgency}` urgency."
    else:
        feedback = f"You incorrectly classified the urgency of the message as `{pred_urgency}`. The correct urgency is `{gold_urgency}`. Think about how you could have reasoned to get the correct urgency label."
    
    return feedback, score

def feedback_sentiment(gold_sentiment, pred_sentiment):
    """Generate feedback for the sentiment module."""
    score = 1.0 if gold_sentiment == pred_sentiment else 0.0
    
    if gold_sentiment == pred_sentiment:
        feedback = f"You correctly classified the sentiment of the message as `{gold_sentiment}`. This message is indeed `{gold_sentiment}`."
    else:
        feedback = f"You incorrectly classified the sentiment of the message as `{pred_sentiment}`. The correct sentiment is `{gold_sentiment}`. Think about how you could have reasoned to get the correct sentiment label."
    
    return feedback, score

def feedback_categories(gold_categories, pred_categories):
    """Generate feedback for the categories module."""
    correctly_included = [k for k, v in gold_categories.items() if v and k in pred_categories]
    incorrectly_included = [k for k, v in gold_categories.items() if not v and k in pred_categories]
    incorrectly_excluded = [k for k, v in gold_categories.items() if v and k not in pred_categories]
    correctly_excluded = [k for k, v in gold_categories.items() if not v and k not in pred_categories]
    
    score = (len(correctly_included) + len(correctly_excluded)) / len(gold_categories)
    
    if score == 1.0:
        fb_text = f"The category classification is perfect. You correctly identified that the message falls under the following categories: `{repr(correctly_included)}`."
    else:
        fb_text = f"The category classification is not perfect. You correctly identified that the message falls under the following categories: `{repr(correctly_included)}`.\n"
        if incorrectly_included:
            fb_text += f"However, you incorrectly identified that the message falls under the following categories: `{repr(incorrectly_included)}`. The message DOES NOT fall under these categories.\n"
        if incorrectly_excluded:
            prefix = "Additionally, " if incorrectly_included else "However, "
            fb_text += f"{prefix}you didn't identify the following categories that the message actually falls under: `{repr(incorrectly_excluded)}`.\n"
        fb_text += "Think about how you could have reasoned to get the correct category labels."
    
    return fb_text, score

def metric_with_feedback(example, pred, trace=None, pred_name=None, pred_trace=None):
    """
    Computes a score and provides feedback text for a specific predictor module.
    Returns a dspy.Prediction with score (float) and feedback (str).
    """
    # Parse gold standard from example
    gold = json.loads(example['answer'])
    
    # Compute feedback and scores for all modules
    fb_urgency, score_urgency = feedback_urgency(gold['urgency'], pred.urgency)
    fb_sentiment, score_sentiment = feedback_sentiment(gold['sentiment'], pred.sentiment)
    fb_categories, score_categories = feedback_categories(gold['categories'], pred.categories)
    
    # Overall score: average of the three accuracies
    total = (score_urgency + score_sentiment + score_categories) / 3
    
    if pred_name is None:
        return total
    elif pred_name == 'urgency_module.predict':
        feedback = fb_urgency
    elif pred_name == 'sentiment_module.predict':
        feedback = fb_sentiment
    elif pred_name == 'categories_module.predict':
        feedback = fb_categories
    
    return dspy.Prediction(score=total, feedback=feedback)
```

## Running GEPA Optimization

```python
from dspy import GEPA

optimizer = GEPA(
    metric=metric_with_feedback,
    auto="light",  # Using light budget for this tutorial
    num_threads=32,
    track_stats=True,
    use_merge=False,
    reflection_lm=dspy.LM(model="gpt-5", temperature=1.0, max_tokens=32000, api_key=api_key)
)

optimized_program = optimizer.compile(
    program,
    trainset=train_set,
    valset=val_set,
)
```

## Optimization Process

```
2025/08/12 20:01:55 INFO: Running GEPA for approx 1643 metric calls of the program. 
                          This amounts to 12.45 full evals on the train+val set.
2025/08/12 20:01:55 INFO: Using 66 examples for tracking Pareto scores.
```

### Iteration 0 (Baseline)
```
2025/08/12 20:01:56 INFO: Average Metric: 47.56666666666666 / 66 (72.1%)
2025/08/12 20:01:56 INFO: Iteration 0: Base program full valset score: 0.7207070707070706
```
- **Baseline valset score: 72.1%**

### Iteration 4 - Urgency Module Optimization
```
2025/08/12 20:01:59 INFO: Average Metric: 52.3 / 66 (79.2%)
2025/08/12 20:01:59 INFO: Iteration 4: New program is on the linear pareto front
2025/08/12 20:01:59 INFO: Iteration 4: Full valset score for new program: 0.7924242424242424
```

**Improved Urgency Module Prompt:**
```
Task: Read the provided message and determine the urgency.

Context/domain:
- Messages typically relate to facility management and services (e.g., facility operations, 
  space utilization, security, sustainability, HVAC systems, maintenance, cleaning services) 
  for a provider like ProCare Facility Solutions.
- Senders may be residential or commercial clients and may reference residents, tenants, 
  property operations, or prior support interactions.

Output format:
- Provide exactly two fields, in this order, no extra text or formatting:
  reasoning: <1–3 concise sentences explaining the key cues that determine urgency>
  urgency: <one of: low | medium | high>

Urgency levels and decision rules:
- HIGH:
  - Clear or implied immediate risk to safety/security or major operational impact.
  - Explicit urgency signals (e.g., "Urgent," "Immediate attention required," "ASAP," 
    "critical," "escalating").
  - Examples: security breach, fire/smoke, flooding, gas leak, electrical hazard, power outage.

- MEDIUM:
  - Time-sensitive issues affecting comfort, reliability, or service quality but not emergencies.
  - Requests for prompt scheduling/repair/maintenance where delay could worsen impact.
  - Examples: HVAC inconsistent temperature, minor malfunction, "at the earliest convenience."

- LOW:
  - General inquiries, information requests, quotes, scheduling discussions with no time pressure.
  - Compliments/feedback without urgent problem.
  - Examples: asking for details on specialized cleaning, pricing, or availability.
```

- **Valset score improved to 79.2%** ✓

### Iteration 5 - Sentiment Module Optimization
```
2025/08/12 20:02:01 INFO: Average Metric: 56.63333333333333 / 66 (85.8%)
2025/08/12 20:02:01 INFO: Iteration 5: New program is on the linear pareto front
2025/08/12 20:02:01 INFO: Iteration 5: Full valset score for new program: 0.8580808080808081
```

**Improved Sentiment Module Prompt:**
```
Task: Read the provided message text and classify its overall sentiment as one of: 
positive, neutral, or negative.

Classification guidelines:
- Focus on the overall emotional tone expressed about the service/interaction, not the 
  message's functional purpose (e.g., making a request) or formalities.
- If signals are mixed or weak, default to neutral.

Label definitions:
- Positive: Clear satisfaction, praise, gratitude, or enthusiasm beyond routine politeness.
  Examples: "satisfied client," "exceptional service," "truly appreciate," "excellent"

- Negative: Dissatisfaction, complaints, frustration, anger, fear, or disappointment.
  Examples: "unacceptable," "unsafe," "poor quality," "frustrated," "very disappointed"

- Neutral: Informational, inquisitive, or routine requests without clear emotional valence.
  Polite language alone does not imply positivity.
  Examples: "I hope this finds you well," "thank you," "best regards"

Output format:
- Output only a single lowercase label: positive, neutral, or negative.
- Do not include any additional text or reasoning.
```

- **Valset score improved to 85.8%** ✓

### Iteration 6 - Categories Module Optimization
```
2025/08/12 20:02:04 INFO: Average Metric: 56.83333333333333 / 66 (86.1%)
2025/08/12 20:02:04 INFO: Iteration 6: New program is on the linear pareto front
2025/08/12 20:02:04 INFO: Iteration 6: Full valset score for new program: 0.861111111111111
```

**Improved Categories Module Prompt:**
```
You are classifying a single customer message sent to ProCare Facility Solutions. 
Your job is to assign all and only the applicable categories from a fixed list, 
based strictly on the message content.

Allowed categories and definitions:
- cleaning_services_scheduling: Primary purpose is to coordinate timing/availability 
  for cleaning services (booking, rescheduling, adjusting times).
  
- specialized_cleaning_services: Mentions specific/specialized cleaning types beyond 
  generic cleaning (deep cleaning, mold remediation, carpet cleaning, window washing).
  
- customer_feedback_and_complaints: Expresses dissatisfaction with prior service, 
  reports subpar outcomes, requests remedies (redo, refund).
  
- quality_and_safety_concerns: Raises quality shortfalls (still dirty/stained) or 
  safety/health risks (mold, hazards).
  
- general_inquiries: Asks for information about services, scope, requirements, or 
  availability before committing.

Key decision rules:
- Multi-label: Assign every category clearly supported by the text.
- Scheduling vs. Complaint: If rescheduling is requested only as a fix within a 
  complaint, do NOT include cleaning_services_scheduling.
- Scheduling vs. Service request: Do NOT tag scheduling merely because sender asks 
  to "arrange a visit" or "send a team."
- Evidence-only: Do not infer intents not supported by the text.

Output format:
- reasoning: Briefly justify each selected category and note key exclusions.
- categories: A JSON array of selected category strings.
```

- **Valset score improved to 86.1%** ✓ (Final best score)

### Final Pareto Front Performance
```
2025/08/12 20:10:06 INFO: Full valset pareto front score: 0.9186868686868687
```
- **Pareto front score: 91.9%** (combining best prompts across all three modules)

## Performance Summary

| Iteration | Module | Valset Score | Improvement | Key Changes |
|-----------|--------|--------------|-------------|-------------|
| 0 (Baseline) | All | 72.1% | - | Original ChainOfThought prompts |
| 4 | Urgency | 79.2% | +7.1% | Added urgency level definitions and decision rules |
| 5 | Sentiment | 85.8% | +6.6% | Clarified sentiment label definitions and edge cases |
| 6 | Categories | 86.1% | +0.3% | Added multi-label logic and scheduling distinctions |
| Final | Pareto Front | 91.9% | +19.8% | Combined best prompts for all modules |

### Optimization Statistics

```
Total metric calls: ~1643
Full evaluations: ~12.45 on train+val set
Training examples: 66
Validation examples: 66
Test examples: 68
Threads: 32
Reflection model: gpt-5
Auto budget: light
```

## Key Insights from GEPA Optimization

1. **Module-Specific Optimization**: GEPA optimized each of the three modules independently, achieving improvements in urgency classification, sentiment analysis, and category assignment.

2. **Structured Feedback**: By providing specific feedback for each module (what was correct, what was incorrect, and why), GEPA could identify precise areas for improvement.

3. **Domain Knowledge Integration**: The optimizer learned domain-specific patterns:
   - Safety/security indicators for high urgency
   - Distinction between polite language and genuine positive sentiment
   - Multi-label classification rules for service categories

4. **Pareto Front Tracking**: GEPA maintained a Pareto front of high-performing variants, allowing different prompts to excel on different subsets of the data.

5. **Substantial Improvement**: Overall improvement from **72.1% → 86.1%** (single best program) and **91.9%** (Pareto front), demonstrating the effectiveness of reflective optimization.

## Conclusion

GEPA successfully optimized a multi-task structured information extraction system by:
- Leveraging execution traces and targeted feedback for each module
- Learning from failure cases to propose improved classification strategies
- Building comprehensive prompt instructions with clear decision rules
- Achieving nearly **20% improvement** in aggregate performance

The optimization process demonstrates GEPA's ability to handle complex multi-module systems where different components require distinct optimization strategies, while maintaining a holistic view of overall system performance.
