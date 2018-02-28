import nltk
import PyPDF2 
import re
import pandas as pd

pdf_file = open('test.pdf','rb') 
read_pdf = PyPDF2.PdfFileReader(pdf_file) 
num_pages = read_pdf.getNumPages()
count = 1
text = ""

#The while loop will read each page
while count < num_pages:
	pageObj = read_pdf.getPage(count) 
	count +=1
	text += pageObj.extractText()
pdf_file.close()

text=re.sub('\\n', '', text)

from nltk.tokenize import sent_tokenize 
sents=sent_tokenize(text)

for i in range(len(sents)):
	sents[i] = sents[i].encode('utf-8')

test = pd.DataFrame()
test['sentence'] = sents
test.to_csv("test.csv",index_label=False)