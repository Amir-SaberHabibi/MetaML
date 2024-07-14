import pandas as pd
import os

results_dir = "src\\results"
file_path = os.path.join(results_dir, "best_result_pso.csv")
df = pd.read_csv(file_path)
print(df)