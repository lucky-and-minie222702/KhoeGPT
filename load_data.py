from datasets import load_dataset
ds = load_dataset("urnus11/Vietnamese-Healthcare") 
ds["vinmec_article_subtitle"].to_pandas().to_csv("train.csv")
ds["medical_qa"].to_pandas().to_csv("test.csv")