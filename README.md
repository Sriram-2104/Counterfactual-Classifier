# Counterfactual-Classifier

## Overview
This project leverages Natural Language Processing (NLP) and machine learning techniques to analyze counterfactual thinking in health-related behavior change. Specifically, it focuses on reflections about blood glucose management and behavioral adjustments for individuals with diabetes. By categorizing participants' counterfactual statements, the system aims to better understand health behavior intentions and enhance personalized interventions.

## Data
The dataset was collected through surveys where participants reflected on elevated blood glucose levels. The survey consisted of:
- **Negative Event Recall**: Participants described situations where their blood glucose exceeded 140 mg/dL.
- **Counterfactual Thinking**: Participants created "If only I...then..." statements to reflect on alternative behaviors they could have performed to improve glucose levels.
- **Follow-Up**: Participants rated the likelihood of applying the counterfactual strategies in the following week.

## Methods
Counterfactual statements were manually labeled using the following criteria:
- **Direction of Counterfactuals**: Classified as upward or downward counterfactuals.
- **Behavioral Intentions**: Key elements were identified to assess participants' potential for behavior change.

## Requirements
- **Python**: Version 3.x
- **Libraries**: 
  - pandas
  - numpy
  - sklearn
  - nltk
  - tensorflow

Install the required libraries using:
```bash
pip install -r requirements.txt
