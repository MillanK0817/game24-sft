# Supervised Fine-tuning of LLMs for the Game of 24

Under the ./game24, it contains the data for the game of 24.
In the train_df.csv, it contains the data for the training set (n = 1262)
In the test_df.csv, it contains the data for the test set (n = 100).

In the ./data, it contains the alpaca format data for direct training.

In the ./evaluations, it contains the responses of different models on the test set and the evaluation results. 

In the ./scripts, it contains the scripts for cot generation, model inference, and evaluation.

Note: the scripts for data visualization are not included, and several data pre-processing are omitted (e.g. from the raw data to the alpaca format)