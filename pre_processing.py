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
            with open (("imovel" + str(count) + ".txt"), "w", encoding = "utf-8") as f2:
                f2.write(text)
        count += 1

def extractCsv(sheet):
    dt = pd.read_csv(
        sheet, sep=',', header=None, 
        names=["Preço", "Latitude", "Longitude", "Quartos","TBD", "Nan2", "Nan3", "Nan4", "Bairro", "Nan5", "Cidade", "Estado", "Url", "Id"])
    #print(dt.shape)
    dt = dt.dropna(axis='columns')
    #print(dt.head())
    return dt

def writeCsv(csv):
    file = "C:\\Users\\mathe\\Documents\\TCC\\csv\\csv_final.csv"
    with open(file, "w", encoding="utf-8") as f1:
        f1.write(csv)    


def main():
    pages = g.glob("C:\\Users\\mathe\\Documents\\TCC\\trindadeimoveis.com.br\\*.html")
    sheets = g.glob("C:\\Users\\mathe\\Documents\\TCC\\trindadeimoveis.com.br\\*.csv")
    #writeTxt(pages)
    dt = pd.DataFrame()
    for sheet in sheets:
        dt = dt.append(extractCsv(sheet))
    csv = dt.to_csv()
    writeCsv(csv)
    


if __name__ == '__main__':
    main()
