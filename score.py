from datasets import load_metric
rouge_metric = load_metric("rouge")

records1 = []
rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]