import pandas as pd
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

# read organizations tagged in articles
data = pd.read_csv('combined_ner_output_org.csv', index_col=0, converters={"intersection_al2": lambda x: x.strip("[]").replace("'", "").split(", ")})
columns = ['article_id', 'organization', 'retrieved_result' ]
retrieved_results = pd.DataFrame(columns=columns)

# load the driver and open corepo
chrome_driver = "C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe"
options = webdriver.ChromeOptions()
options.add_experimental_option('excludeSwitches', ['enable-logging'])
# options.add_argument("user-agent='xyz'")
driver = webdriver.Chrome(executable_path=chrome_driver, options=options)
driver.maximize_window()
driver.get("https://corepo.org/")

# go to some random search page for consistency
driver.find_element_by_id('main-search-box').send_keys('test')
driver.find_element_by_class_name('logo-red').click()
time.sleep(5) # it takes some time to load results
# driver.implicitly_wait(5)

for i in range(2, 10):
    article_id = data.iloc[i]['article_id']
    print(i, end=" ")
    for org in data.iloc[i]['intersection_al2']:
        j = 0

        while j<10:
            j += 1
            try:
                driver.find_element_by_id('search-field').clear()
                driver.find_element_by_id('search-field').send_keys(org)
                driver.find_element_by_id('search-field').send_keys("\n")

                time.sleep(10)

                page_source = driver.page_source #driver.find_element_by_tag_name('body')
                retrieved_result = BeautifulSoup(page_source, 'lxml')
                retrieved_result = retrieved_result.find('body').find('div', {'class':'flex-1 mycontainer'})

                # if container can't be found
                if retrieved_result == None:
                    retrieved_result = "Empty"
                # retrieved_result = str(retrieved_result).replace("/n", " ")
                # print(retrieved_result.prettify())
                temp = pd.DataFrame([[article_id, org, retrieved_result]], columns=columns)
                # retrieved_results = pd.concat([retrieved_results, temp], ignore_index=True)

                temp.to_csv('corepo_lookup_dump.csv', encoding='utf-8-sig', mode='a', quotechar='~', header=None)
                break
            except:
                print("Error at " + str(i) + org )
    print("")

# close the browser after 45 seconds
time.sleep(10)
driver.close()
# retrieved_results.to_csv('corepo_lookup_dump.csv', encoding='utf-8-sig', mode='a', header=False)