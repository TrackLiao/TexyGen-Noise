#!/usr/bin/env python
# coding=utf-8
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()

    return text
def extract_data(html) : 
    text = []

    for item in html.split("</review_text>"):
        if "<review_text>" : 
            text.append(item[item.find("<review_text")+len("<review_text>") : ])
    print("number of text : ", len(text))
    return text

def save_doc(text, filename):
    file = open(filename, 'w')
    file.write('\n'.join(text))
    file.close

# infile = 'cellphone'
# infile = 'electronic'
infile = 'all'

html = load_doc(infile)
text = extract_data(html)
save_doc(text, 'data.txt')



