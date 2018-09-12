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
    csv = pd.read_csv(
        sheet, sep=',', header=None, 
        names=["Preço", "Latitude", "Longitude", "Quartos", "Bairro", "Cidade", "Estado", "Url", "Id"])
    #print(csv.shape)
    print(csv.head())
    return csv
        


def main():
    pages = g.glob("C:\\Users\\mathe\\Documents\\TCC\\trindadeimoveis.com.br\\*.html")
    sheets = g.glob("C:\\Users\\mathe\\Documents\\TCC\\trindadeimoveis.com.br\\*.csv")
    #writeTxt(pages)
    for sheet in sheets:
        csv = extractCsv(sheet)


if __name__ == '__main__':
    main()
