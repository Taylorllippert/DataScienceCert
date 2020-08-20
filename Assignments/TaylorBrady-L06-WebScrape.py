# File: TaylorBrady-L06-WebScrape.py
# Description: Counts https links contained in news article
# Author: Taylor Brady
# Date: 7/17/2020

import requests
from bs4 import BeautifulSoup as bs
import re


if __name__ == "__main__":
    # Pull and store html content from url in soup object
    url = "https://www.nytimes.com/2020/08/11/us/politics/kamala-harris-vp-biden.html"
    response = requests.get(url)
    content = response.content
    soup = bs(content, 'lxml')

    # Find all elements within the a tag where the href begins with https
    all_a = soup.find_all('a', href=re.compile("^https"))
    # Display size of all_a
    print("Number of links found: " + str(all_a.__len__()))
