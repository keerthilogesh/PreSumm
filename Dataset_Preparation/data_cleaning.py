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

##
ip_data["content"] = ip_data["content"].apply(lambda x: str(x).replace("\n", " ").replace(". ", ".\n"))
ip_data["caption"] = ip_data["caption"].apply(lambda x: str(x).replace("\n", " ").replace(". ", ".\n"))
ip_data["content_length"] = ip_data["content"].apply(lambda x: len([xx for xx in x.split(".\n") if len(xx) > 50]))  # number of sentences with num of characters > 50
ip_data["word_length"] = ip_data["content"].apply(lambda x: [len(xx.split(" ")) for xx in x.split(".\n") if len(xx) > 50])  # number of sentences with num of characters > 50
ip_data["word_length_sum"] = ip_data["word_length"].apply(lambda x: sum(x))
import hashlib
def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))  # encode() : Converts the string into bytes to be acceptable by hash function.
    return h.hexdigest()  # hexdigest() : Returns the encoded data in hexadecimal format.

ip_data["docid_story"] = ip_data["docid"].apply(lambda x: hashhex(x.strip())+ ".story")


## clean and tokenize subset of data
ip_data_n = ip_data.copy()
ip_data_n = ip_data_n[ip_data_n["caption"].str.len() > 10]
ip_data_n = ip_data_n[ip_data_n["content"].str.len() > 100]
ip_data_n = ip_data_n[(ip_data_n["content_length"] >= 15) & (ip_data_n["content_length"] <= 150)]
ip_data_n = ip_data_n.reset_index(drop=True)
ip_data_n = ip_data_n[ip_data_n["n_pages"] < 10]
ip_data_n["content"] = ip_data_n["content"].apply(lambda contents: ".\n".join(
    [re.sub('\s+', ' ', line) for line in contents.strip().replace("  ", " ").split(".\n") if len(re.sub('\s+', ' ', line).split(" ")) > 5]))

##
# ip_data_n = ip_data_n[ip_data_n["n_pages"] < 5]
# ip_data_n["caption_clean"] = ip_data_n["caption"].apply(lambda x: preprocess_sent(str(x).replace("\n", " ")))
# ip_data_n["content_clean"] = ip_data_n["content"].apply(lambda x: preprocess_sent(str(x).replace("\n", " ")))
# ip_data_n["caption_token"] = ip_data_n["caption_clean"].apply(lambda x: preprocess_word(x))
# ip_data_n["content_token"] = ip_data_n["content_clean"].apply(lambda x: preprocess_word(x))
#ip_data_n["caption_token"] = ip_data_n["caption_token"].apply(lambda x: " ".join(x))
#ip_data_n["content_token"] = ip_data_n["content_token"].apply(lambda x: " ".join(x))
ip_data_n = ip_data_n.reset_index(drop=True)

##
ip_data_n.to_excel(r"C:\Users\keert\PycharmProjects\PreSumm\Dataset_Preparation\cleaned_data\tokenized_data3.xlsx", engine='xlsxwriter', index=False)

##
# ip_data_n[["docid", "caption_token", "content_token"]].to_json(r"C:\Users\keert\PycharmProjects\PreSumm\Dataset_Preparation\cleaned_data\tokenized_data3.json", orient="index")

