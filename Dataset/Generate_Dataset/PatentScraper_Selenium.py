'''
Code to scrape all english abstracts 
From links in downloaded XLSX files
Note - to use download file, need linux, if not use other scraper.
'''

import numpy as np
import pandas as pd
import glob
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver import FirefoxOptions
import os
import itertools
import random
import csv

#----Path to selenium driver, gecko for Firefox----#
spath = ''
os.environ['PATH'] += os.pathsep + spath

# XLSX to pandas dataframe
def read_xlsx(fn):
    # format the xlsx dataframe 
    df = pd.read_excel(fn, header = None)
    df = df.drop([0])
    headers = df.iloc[0]
    df.columns = headers
    df = df.drop([1])
    df = df.reset_index()

    return df

# List of urls from dataframe
def get_enurl(df):
    urls = []
    urlnum = 0
    for i in range(0, len(df['result link'])):
        if urlnum < 10000:
            try:
                lang = (df['result link'][i].split('/')[-1])
            except AttributeError:
                continue
            if lang == 'en':
                urls.append(df['result link'][i])
            urlnum += 1
    return urls

# Scrapes section needed, i.e. abstract or claims or description
def get_section(urls, classpath, savefile):
    opts = FirefoxOptions()
    opts.add_argument("--headless")
    driver = webdriver.Firefox(options=opts)
    
    head = ['Abstract', 'Claim', 'Description', 'GreenV']
    f = open(savefile, 'a')
    writer = csv.writer(f)
    writer.writerow(head)
    f.close()
    
    num = 0
    for url in urls:
        print(num)
        if num < 100000:
            section = ['0', '0','0', '0']
            driver.get(url)
            for i in range(0, len(classpath)):
                #xpath for selenium
                sec = driver.find_elements(by=By.XPATH, value ='//div[@class = "{}"]'.format(classpath[i]))
                if len(sec) == 0:
                    section[i] = ('NaN')
                else:
                    section[i] = (sec[0].text)
            f = open(savefile, 'a')
            writer = csv.writer(f)
            # Write section to file as scraping
            writer.writerow(section)
            f.close()
            num += 1
    return

# XLSX files are saved as gp-search{}
path = glob.glob('/path/to/gp-search*')

allurls = []
for i in path:
    df = read_xlsx(i)
    urls = get_enurl(df)
    allurls.append(urls)

#turn into flat list
allurls = list(itertools.chain(*allurls))

#list of things to get:
secs = ["abstract style-scope patent-text", "claim style-scope patent-text", "description style-scope patent-text"]
print(len(allurls))

get_section(allurls, secs, savefile)

