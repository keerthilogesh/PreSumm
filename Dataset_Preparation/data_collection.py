import numpy as np
import os
import PyPDF2
import pandas
from bs4 import BeautifulSoup

data_dir = "C:/Users/keert/PycharmProjects/PreSumm/Dataset_Preparation/Downloads_PDF/"
all_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if ".pdf" in file]

all_data = []
for pdfFile in all_files:
    try:
        ##
        # pdfFile = "C:/Users/keert/PycharmProjects/PreSumm/Dataset_Preparation/Verified_PDF_File/0a17c7df-6d8c-41f7-8ac8-0c9774b8d3a1.pdf"
        # pdfFile = "C:/Users/keert/PycharmProjects/PreSumm/Dataset_Preparation/Verified_PDF_File/0a163cae-52f6-4142-bbfe-4e1e174a35c4.pdf"
        # pdfFile = "C:/Users/keert/PycharmProjects/PreSumm/Dataset_Preparation/Verified_PDF_File/0a72ade2-c5b0-4946-80a6-5f61fd357820.pdf"
        # pdfFile = "C:/Users/keert/PycharmProjects/PreSumm/Dataset_Preparation/Verified_PDF_File/16e4ca6c-0ed7-47dc-9def-8447c1e7d655.pdf"
        # pdfFile = "C:/Users/keert/PycharmProjects/PreSumm/Dataset_Preparation/Downloads_PDF/14a0b2c8-8c4e-4acf-b4c3-742e0a2e1bb6.pdf"
        # pdfFile = "C:/Users/keert/PycharmProjects/PreSumm/Dataset_Preparation/Downloads_PDF/d4904999-b567-42f8-bd25-5f9411c0097b.pdf"
        # print(f"processing file: {pdfFile}")
        pdfRead = PyPDF2.PdfFileReader(pdfFile) # read PDF file
        index_page_content = BeautifulSoup(pdfRead.pages[0].extractText(), 'html.parser').getText()  # Extracting the Page 1
        docid = os.path.basename(pdfFile).split(".pdf")[0]   # It returns the tail of the path i.e DOI
        n_pages = pdfRead.numPages
        if "Caption" in index_page_content:
            if "  " in index_page_content.split("Caption")[0].replace("\n", " ").strip():
                title = index_page_content.split("Caption")[0].replace("\n", " ").strip().split("  ")[1].strip()  # Strip()- removes the whitespaces from the beginning and the end . Split() using the word 'Caption' and extract the title
            else:
                title = index_page_content.split("Caption")[0].replace("\n", " ").strip()
            title_inside = list(pdfRead.pages)[1:][0].extractText()[6:].split("\n")[:1]  # to extract the title from the second page
            caption = index_page_content.split("Caption:")[1].split("Source")[0].replace("\n", " ").strip()  # Extracting only the caption from the page 1
            content = ""
            for page in list(pdfRead.pages)[1:]:
                content = content + "\n" + "\n".join(page.extractText().split("\n")[1:]).strip()  # Extracting the content
            # print(f"title: {title}, caption: {caption}, content: {content}")
            all_data.append({"docid": docid, "title": title, "inside_title": title_inside, "caption": caption,
                             "content": content, "n_pages": n_pages})
        else:
            # print(f"No caption for file: {pdfFile}")
            if "  " in index_page_content.split("Source")[0].replace("\n", " ").strip():
                title = index_page_content.split("Source")[0].replace("\n", " ").strip().split("  ")[1].strip()
            else:
                title = index_page_content.split("Source")[0].replace("\n", " ").strip()
            title_inside = list(pdfRead.pages)[1:][0].extractText()[6:].split("\n")[:1]
            content = ""
            for page in list(pdfRead.pages)[1:]:
                content = content + "\n" + "\n".join(page.extractText().split("\n")[1:]).strip()
            # print(f"title: {title}, caption: {caption}, content: {content}")
            all_data.append({"docid": docid, "title": title, "inside_title": title_inside, "caption": np.nan,
                             "content": content, "n_pages": n_pages})
    except:
        print(f"Exception for file: {pdfFile}")

all_data = pandas.DataFrame(all_data)
all_data.to_excel(r"C:\Users\keert\PycharmProjects\PreSumm\Dataset_Preparation\cleaned_data\cleaned_data.xlsx", engine='xlsxwriter', index=False)
all_data.to_pickle(r"C:\Users\keert\PycharmProjects\PreSumm\Dataset_Preparation\cleaned_data\cleaned_data.pkl")
##


