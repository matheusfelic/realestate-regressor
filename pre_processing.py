from bs4 import BeautifulSoup as bs
import glob as g
import sys
import pandas as pd
import numpy as np

def extractText(html):
    soup = bs(html, 'html.parser')
    for script in soup(["script", "style"]):
        script.extract()
    text = soup.getText().lower()
    return text

def writeTxt(pages):
    count = 1
    for file in pages:
        with open(file, "r", encoding="utf-8", errors="ignore") as f1:
            text = extractText(f1)
            #print("finishing extracting...")
            if(count < (len(pages) * 0.8)):
                with open (("poa_sample/txts/train/poa_realestate" + str(count) + ".txt"), "w", encoding = "utf-8") as f2:
                    f2.write(text)
                    #print("finishing writing...")
            else:
                with open (("poa_sample/txts/test/poa_realestate" + str(count) + ".txt"), "w", encoding = "utf-8") as f2:
                    f2.write(text)
        count += 1

def extractCsv(sheet):
    dt = pd.read_csv(
        sheet, sep=',', header=None, 
        names=["price", "latitude", "longitude", "bedrooms", "bathrooms", "area", "pkspaces", "ensuites", "timestamp", "type", "operation", "url"])
    #print(dt.shape)
    dt = dt.dropna(axis='columns')
    #print(dt.head())
    return dt

def writeCsv(csv, type):
    file = "RJ_final_"+ type +".csv"
    with open(file, "w", encoding="utf-8") as f1:
        f1.write(csv) 

def orderPages(dt1, pages):
    p = []
    urls = dt1['url'].tolist()
    #print(urls)
    for url in urls:    
        for page in pages:
            stripped_page = page[11:-5] #para poa
            #stripped_page = page[10:-5] # para sp e rj
            #print(stripped_page)
            if stripped_page == url:
                #print("ordered")
                p.append(page)
                break  
    return p  

def main():
    pages = g.glob("poa_sample/*.html")
    #print(pages)
    dt1 = extractCsv("POA_sample.csv")
    #print(sheets)
    dt1 = dt1[dt1.operation != 'rent']
    pages = orderPages(dt1, pages)
    print("finishing ordering...")
    writeTxt(pages)
    dt2 = pd.DataFrame()
    cond = dt1.index < (len(dt1) * 0.8)
    dt2 = dt1.loc[cond, :]
    dt1.drop(dt2.index, inplace=True)
    csv1 = dt2.to_csv(index=False)
    csv2 = dt1.to_csv(index=False)
    #writeCsv(csv1, "train")
    #writeCsv(csv2, "test")
    


if __name__ == '__main__':
    main()
