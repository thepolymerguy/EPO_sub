import time
import csv
import subprocess
import pandas as pd

'''
Downloads XLSX file with all the links of patents associated to class
Note this download method works for Linux/ machines that use bash
runcmd function can be modified if using Windows
'''

#--------- function to handle the subprocess --------#
#---- not sure if this will run on windows ----#
def runcmd(cmd, verbose = False, *args, **kwargs):
    # Allows python to run a bash command
    # Makes sure the error of the subproceess is communicated if it fails
    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass

#--------- function to change url and download file --------#
def download_csv(fn):

    df = pd.read_csv(fn)
    # Reads in classes into data frame
    classname = df.loc[:, 'Class'].values.flatten().tolist()

    for i in classname:
        print(i)
        c = i.split('/')
        url = 'https://patents.google.com/xhr/query?url=q%3D{a}%252f{b}%26oq%3D{a}%252f{b}&exp=&download=true&download_format=xlsx'.format(a=c[0], b=c[1])
        ##url1 = 'https://patents.google.com/xhr/query?url=q%3D{a}%26oq%3D{a}&exp=&download=true&download_format=xlsx'.format(a=c[0])
        # If needed url1 to download csv for the root class (not the branches)
        runcmd('firefox "{}"'.format(url), verbose = True)
        # Note url needs to be in " " otherwise will just open json and not download file
        time.sleep(20)
        # Sleep to stop error from google patents (too many requests)
        # Use runcmd to move file from downloads to working directory

download_csv('/path/to/MainClassDescriptions_cleaned.csv')
download_csv('/pth/to/GreenClassDescriptions_cleaned.csv')
# CSV file with all the classes to download

