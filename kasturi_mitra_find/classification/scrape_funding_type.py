import time

import requests
import pandas as pd
from bs4 import BeautifulSoup

corepo_orgs = pd.read_csv('../evaluation/corepo_lookup_preprocessed.csv', index_col=0)
cb_data = pd.read_csv('../evaluation/selected_organizations_crunchbase.csv', index_col=0)

headers = {'User-Agent': 'Mozilla/5.0'}

corepo_orgs['last_funding_type'] = None
for i in range(0,20):
    print(i, end=' ')
    time.sleep(10)
    scraped_org = corepo_orgs.iloc[i]
    url = str(scraped_org['domain'])
    if url == 'nan':
        continue
#     print(scraped_org)
#     print(url)
    url = url.split("://")[1]
    url1 = "://" + url
    url2 = "." + url
    #     print(url1, url2)
    # to avoid partial matches, make sure match url has www.url or http(s)://url
    temp = cb_data[cb_data['homepage_url'].str.contains(url1, na=False, regex=False) | cb_data['homepage_url'].str.contains(url2, na=False, regex=False)]
    if temp.empty:
        pass
    else:
        temp = temp.iloc[0]
        # corepo_orgs['last_funding_type'].iloc[i] = temp['cb_url']
        cb_url = temp['cb_url']
        page = requests.get(cb_url, headers=headers)
        soup = BeautifulSoup(page.content, 'html.parser')
        attrs = soup.find('ul', class_='text_and_value')
        value = ""
        for attr in attrs:
            text = attr.text
            if text.find('Last Funding Type')!=-1:
                value = text.replace('Last Funding Type', '').strip()
                print(value)
        corepo_orgs['last_funding_type'].iloc[i] = value
        print('')

corepo_orgs.to_csv('corepo_lookup_funding_type.csv', encoding='utf-8-sig')