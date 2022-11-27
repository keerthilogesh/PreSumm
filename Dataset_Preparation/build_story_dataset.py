"""
Aim: The aim of this file is to generate gold summary based on the caption.
"""
import os
import pandas
import hashlib

ip_data = pandas.read_excel(
    r"C:\Users\keert\PycharmProjects\PreSumm\Dataset_Preparation\cleaned_data\tokenized_data3.xlsx", engine="openpyxl")
op_data_dir = "C:/Users/keert/PycharmProjects/PreSumm/data/historical_samples/raw_stories/"
uri_data_dir = "C:/Users/keert/PycharmProjects/PreSumm/data/historical_samples/urls/"

##

captions, contents, filenames = ip_data["caption"].values, ip_data["content"].values, ip_data["docid"].values


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))  # encode() : Converts the string into bytes to be acceptable by hash function.
    return h.hexdigest()  # hexdigest() : Returns the encoded data in hexadecimal format.


to_writes = []
for caption, content, fn in zip(captions, contents, filenames):
    to_write = fn
    print(fn, end=" - ")
    fn = os.path.join(op_data_dir, hashhex(fn.strip()) + ".story")
    print(fn)
    to_write = to_write + " - " + fn
    to_writes.append(to_write)
    with open(fn, "w", encoding="utf-8") as file:
        file.write(content)
        file.write("\n\n@highlight\n\n")
        caption = caption.strip().split(".")
        caption = [c for c in caption if len(c) > 10]
        caption = "\n\n@highlight\n\n".join(caption)
        file.write(caption)

with open("C:/Users/keert/PycharmProjects/PreSumm/data/hist_samples_map.csv", "w", encoding="utf-8") as file:
    for to_write in to_writes:
        file.write(to_write)
        file.write("\n")
    file.close()

## train test validation split
import random

random.seed(000000)
docids = ip_data["docid"].tolist()
random.shuffle(docids)
docids_test = docids[:int(len(docids) * 0.05)]
docids_valid = docids[int(len(docids) * 0.05):2 * int(len(docids) * 0.05)]
docids_train = docids[2 * int(len(docids) * 0.05):]

os.makedirs(uri_data_dir, exist_ok=True)
with open(os.path.join(uri_data_dir, "mapping_train.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(docids_train))
    f.close()
with open(os.path.join(uri_data_dir, "mapping_test.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(docids_test))
    f.close()
with open(os.path.join(uri_data_dir, "mapping_valid.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(docids_valid))
    f.close()

##
ip_data[["docid", "fn"]].to_csv(r"C:\Users\keert\PycharmProjects\PreSumm\data\hist_samples_map.csv", index=False)
