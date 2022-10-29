import sys
import numpy as np
import pandas as pd
from Dataset_Preparation.preprocess import preprocess_sent, preprocess_word
from dateparser.search import search_dates
import re

ip_data = pd.read_excel(r"C:\Users\keert\PycharmProjects\PreSumm\Dataset_Preparation\cleaned_data\cleaned_data.xlsx", engine="openpyxl")  # reading the excel file (cleaned data)
ip_data["location_date"] = ip_data["title"].apply(lambda x: re.findall(r"\((.*?)\)", str(x))[-1] if len(re.findall(r"\((.*?)\)", str(x))) > 0 else "")
ip_data["location_date"] = ip_data["location_date"].apply(lambda x: x if len(str(x)) > 3 else "")
ip_data["date"] = ip_data["location_date"].apply(lambda x: x.split(",")[1] if "," in x else x)
ip_data["location"] = ip_data["location_date"].apply(lambda x: x.split(",")[0] if "," in x else None)

## clean and tokenize subset of data
ip_data_n = ip_data.copy()
ip_data_n = ip_data_n[ip_data_n["caption"].str.len() > 10]
ip_data_n = ip_data_n[ip_data_n["content"].str.len() > 100]
ip_data_n = ip_data_n[ip_data_n["title"].str.len() > 10]
ip_data_n = ip_data_n[ip_data_n["n_pages"] < 5]
ip_data_n["caption_clean"] = ip_data_n["caption"].apply(lambda x: preprocess_sent(str(x).replace("\n", " ")))
ip_data_n["content_clean"] = ip_data_n["content"].apply(lambda x: preprocess_sent(str(x).replace("\n", " ")))
ip_data_n["caption_token"] = ip_data_n["caption_clean"].apply(lambda x: preprocess_word(x))
ip_data_n["content_token"] = ip_data_n["content_clean"].apply(lambda x: preprocess_word(x))
#ip_data_n["caption_token"] = ip_data_n["caption_token"].apply(lambda x: " ".join(x))
#ip_data_n["content_token"] = ip_data_n["content_token"].apply(lambda x: " ".join(x))
ip_data_n = ip_data_n.reset_index(drop=True)

##
ip_data_n.to_excel(r"C:\Users\keert\PycharmProjects\PreSumm\Dataset_Preparation\cleaned_data\tokenized_data3.xlsx", engine='xlsxwriter', index=False)

##
ip_data_n[["docid", "caption_token", "content_token"]].to_json(r"C:\Users\keert\PycharmProjects\PreSumm\Dataset_Preparation\cleaned_data\tokenized_data3.json", orient="index")

##
