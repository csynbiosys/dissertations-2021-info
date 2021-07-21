from bs4 import BeautifulSoup
import pandas as pd


data = pd.read_csv("corepo_lookup_dump.csv", quotechar='~', index_col=False)
columns = ['article_id', 'search_keyword', 'name', 'description', 'domain', 'founding_year', 'funding', 'employee_count', 'industry', 'location', 'city', 'country', 'linkedin']
structured_data = pd.DataFrame(columns=columns)
# print(data.shape)

for i in range(0, len(data)):
    print(i)
    article_id = data.iloc[i]['article_id']
    text = data.iloc[i]['retrieved_result']
    org = data.iloc[i]['organization']
    if text != 'Empty':
        soup = BeautifulSoup(text, 'lxml')
        results = soup.find('div', id='result-list')
        if (results == None) or (results.text.strip()==""):
            print("no results")
        else:
            results = results.findAll('div', class_='result add-to-list-bearer')
            for result in results:
                title = result.find('h4', class_='result-title')
                if title != None:
                    title = title.text.strip()
                else:
                    title = ""

                description = result.find('p', class_='result-description')
                if description != None:
                    description = description.text.strip().replace('\n', '')
                else:
                    description = ""

                domain = result.find('p', class_='result-domain')
                if domain != None:
                    domain = domain.text.strip()
                else:
                    domain = ""

                founding_year = funding = employee_count = industry = location = city = country = linkedin = ""
                details = result.findAll('div', class_='w-full md:w-1/2 mb-1')
                for detail in details:
                    text = detail.text.strip()
                    text = text.split(":")
                    if text[0] == "Founded":
                        founding_year = text[1].strip()
                    elif text[0] == "Funding":
                        temp = detail.find('input', {'type':'submit'})
                        funding = temp['value'].strip()
                    elif text[0] == "Employee Count":
                        temp = detail.find('input', {'type': 'submit'})
                        employee_count = temp['value'].strip()
                    elif text[0] == "Industry":
                        industry = text[1].strip()
                    elif text[0] == "Location":
                        temp = text[1].strip()
                        temp = temp.split(",")
                        # for t in temp:
                        #     if t.strip() != "":
                        #         location = location + t.strip() + ", "
                        city = temp[0].strip()
                        country = temp[1].strip()
                        location = city + ", " + country
                    elif text[0] == "Social":
                        temp = detail.find('a')
                        linkedin = temp['href']

                # print(title)
                # print(description)
                # print(domain)
                # print(founding_year)
                # print(funding)
                # print(employee_count)
                # print(industry)
                # print(location)
                # print(city)
                # print(country)
                # print(linkedin)
                temp = pd.DataFrame([[article_id, org, title, description, domain, founding_year, funding, employee_count, industry, location, city, country, linkedin]], columns=columns)
                structured_data = pd.concat([structured_data, temp], ignore_index=True)

structured_data.to_csv('corepo_lookup_structured.csv', encoding='utf-8-sig')