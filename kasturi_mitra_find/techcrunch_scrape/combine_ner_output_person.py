import pandas as pd

# load named entities from three sources
spacy = pd.read_csv("articles_entities_spacy.csv", index_col=0, converters={"people": lambda x: x.strip("[]").split(", ")})
stanford = pd.read_csv("articles_entities_stanford.csv", index_col=0, converters={"people": lambda x: x.strip("[]").split(", ")})
nltk = pd.read_csv("articles_entities_nltk.csv", index_col=0, converters={"people": lambda x: x.strip("[]").split(", ")})

columns = ['article_id', 'union', 'intersection', 'intersection_al2']
data = pd.DataFrame(columns=columns)

for i in range(0, len(spacy)):
    print(i)
    union = []
    #
    # print(str(spacy.iloc[i]['people']) + "\n" + str(stanford.iloc[i]['people']) + "\n" + str(nltk.iloc[i]['people']))

    # union of potential startups
    union.extend(spacy.iloc[i]['people'])
    union.extend(stanford.iloc[i]['people'])
    union.extend(nltk.iloc[i]['people'])
    # remove duplicates
    union = list(dict.fromkeys(union))

    # remove extra '
    for x in range(len(union)):
        union[x] = union[x].replace("'", "")
    # remove any empty elements in the list
    union = list(filter(None, union))
    # print(union)

    # intersection of potential startups
    intersection1 = list(set(spacy.iloc[i]['people']) & set(stanford.iloc[i]['people']))
    intersection2 = list(set(stanford.iloc[i]['people']) & set(nltk.iloc[i]['people']))
    intersection3 = list(set(nltk.iloc[i]['people']) & set(spacy.iloc[i]['people']))

    # intersetion of SpaCy, Stanford, NLTK
    intersection = list(set(intersection1) & set(nltk.iloc[i]['people']))
    # intersection with orgs appearing in any two
    intersection_at_least_two = []
    intersection_at_least_two.extend(intersection1)
    intersection_at_least_two.extend(intersection2)
    intersection_at_least_two.extend(intersection3)
    # remove duplicates
    intersection_at_least_two = list(dict.fromkeys(intersection_at_least_two))

    # remove extra '
    for x in range(len(intersection)):
        intersection[x] = intersection[x].replace("'", "")
    for x in range(len(intersection_at_least_two)):
        intersection_at_least_two[x] = intersection_at_least_two[x].replace("'", "")

    # print(intersection_at_least_two)
    # print(intersection)

    temp = pd.DataFrame([[i, union, intersection, intersection_at_least_two]], columns=columns)
    data = pd.concat([data, temp], ignore_index=True)


# save locally
data.to_csv('combined_ner_output_person.csv', encoding='utf-8-sig')