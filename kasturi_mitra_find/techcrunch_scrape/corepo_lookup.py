import time

import requests
import pandas as pd

# read organizations tagged in articles
data = pd.read_csv('combined_ner_output_org.csv', index_col=0, converters={"intersection_al2": lambda x: x.strip("[]").replace("'", "").split(", ")})
columns = ['article_id', 'organization', 'retrieved_result' ]
retrieved_results = pd.DataFrame(columns=columns)

for i in range(0, 10):
    article_id = data.iloc[i]['article_id']
    print(i, end=" ")
    for org in data.iloc[i]['intersection_al2']:
        j = 0

        while j<10:
            j += 1
            try:
                time.sleep(3)
                retrieved_result = ""
                query = {'q': org, 'page_size': '25'}

                resp = requests.get('https://corepo.org/api/v1/search/', params=query)
                # if request is not successful
                if str(resp) != '<Response [200]>':
                    print(resp.json(), resp)
                    if str(resp) == '<Response [429]>':
                        time.sleep(100)
                        continue
                    else:
                        retrieved_result = "error"
                else:
                    retrieved_result = resp.json()

                temp = pd.DataFrame([[article_id, org, retrieved_result]], columns=columns)
                retrieved_results = pd.concat([retrieved_results, temp], ignore_index=True)
                break
            except:
                print("Error at " + str(i) + org )
    print("")

retrieved_results.to_csv('corepo_lookup_dump.csv', encoding='utf-8-sig')