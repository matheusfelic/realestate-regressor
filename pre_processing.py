from bs4 import BeautifulSoup as bs
import glob as g
import sys

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

#def extractCsv(sheets):
    #for csv in sheets:


def main():
    pages = g.glob("C:\\Users\\mathe\\Documents\\TCC\\trindadeimoveis.com.br\\*.html")
    sheets = g.glob("C:\\Users\\mathe\\Documents\\TCC\\trindadeimoveis.com.br\\*.csv")
    #writeTxt(pages)
    extractCsv(sheets)


if __name__ == '__main__':
    main()
