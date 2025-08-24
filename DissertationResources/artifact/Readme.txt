This is a simple Python script (ml_model_training.py) to classify network traffic. The goal is to figure out if network activity is normal (Benign) or if it's a command-and-control (C2) attack. It is a full machine learning pipeline, so it does everything from preparing the data to training models and showing the final results.

Link to external dataset : https://data.mendeley.com/datasets/wjxc69xj3n/1
Link to synthetic dataset : https://github.com/parthbhagat1337/C2-Havoc-Synthetic-dataset.git
==================================================================================================================

How It Works

It's a step-by-step process. Here's what the code does:

[+] Loads Datasets: It reads in two CSV files—one for training and one for testing.

[+] Cleans Data: It cleans up the data by getting rid of text columns and fixing any weird values like "infinity" or empty cells.

[+] Splits Data: It splits the big training dataset into two parts: one for actual training and a smaller part to check the model's performance on unseen data (I call this the validation set).

[+] Scales Features: It makes sure all the numbers are on a similar scale so the models can learn better.

[+] Handles Imbalance: Since malicious traffic is rare, I used a technique called SMOTE to create some fake malicious samples. This helps the models learn without being biased toward the normal traffic.

[+] Trains Models: It trains three popular models: Logistic Regression, Random Forest, and XGBoost.

[+] Evaluates Models: It runs the trained models on both the validation set and a completely separate external test set to see how well they did.

[+] Saves Everything: It creates a folder and saves all the results, charts, and a summary report so I can look at them later.

==================================================================================================================

How to Run This Code

[+] Get the Files: Put the ml_model_training.py script and your two CSV dataset files in the same folder.

[+] Change the Filenames: Open the ml_model_training.py file. Scroll all the way down to the bottom. You'll see two lines that look like this:

train_dataset_path = "Synthetic-Havoc-Dataset.csv"
test_dataset_path = "External-C2-Dataset.csv"

You need to change the names in the quotes to the exact names of your training and testing CSV files.

[+] Install the Libraries: You'll need some extra libraries to make the script work. The easiest way is to use the requirements.txt file I made. Open your terminal or command prompt, go to the folder where you saved the files, and type this command:

pip install -r requirements.txt

[+] Now you're ready to run it. Just type this into your terminal:

python3 ml_model_training.py

==================================================================================================================

What You'll Get

When you run the script, a new folder called ml_results will appear. Inside, you'll find:

[+] results_summary.txt: A text file that gives you all the final scores for each model (like Accuracy, F1-score, etc.).

[+] confusion_matrices.png: An image with a chart showing how many predictions were correct and how many were wrong.

[+] learning_curves.png: A chart that shows how the models' performance changed as they saw more data. It helps me see if the model is getting better or if it's stuck.

[+] feature_importance.png: A bar chart that shows which data features (like "number of packets" or "port number") were most important for the models.

[+] Other .csv files: There will be some extra CSV files that contain the raw data for feature importance.

You'll also see all these results and some simple charts printed directly in your terminal as the script runs.
