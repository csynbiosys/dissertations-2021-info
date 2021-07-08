import pandas as pd

# load named entities from three sources
spacy = pd.read_csv("articles_entities_spacy.csv", index_col=0, converters={"potential_startups": lambda x: x.strip("[]").split(", ")})
stanford = pd.read_csv("articles_entities_stanford.csv", index_col=0, converters={"potential_startups": lambda x: x.strip("[]").split(", ")})
nltk = pd.read_csv("articles_entities_nltk.csv", index_col=0, converters={"potential_startups": lambda x: x.strip("[]").split(", ")})

columns = ['article_id', 'union', 'intersection', 'intersection_al2']
data = pd.DataFrame(columns=columns)

for i in range(0, len(spacy)):
    print(i)
    union = []
    #
    # print(str(spacy.iloc[i]['potential_startups']) + "\n" + str(stanford.iloc[i]['potential_startups']) + "\n" + str(nltk.iloc[i]['potential_startups']))

    # union of potential startups
    union.extend(spacy.iloc[i]['potential_startups'])
    union.extend(stanford.iloc[i]['potential_startups'])
    union.extend(nltk.iloc[i]['potential_startups'])
    # remove duplicates
    union = list(dict.fromkeys(union))

    # remove extra '
    for x in range(len(union)):
        union[x] = union[x].replace("'", "")
    # remove any empty elements in the list
    union = list(filter(None, union))
    # print(union)

    # intersection of potential startups
    intersection1 = list(set(spacy.iloc[i]['potential_startups']) & set(stanford.iloc[i]['potential_startups']))
    intersection2 = list(set(stanford.iloc[i]['potential_startups']) & set(nltk.iloc[i]['potential_startups']))
    intersection3 = list(set(nltk.iloc[i]['potential_startups']) & set(spacy.iloc[i]['potential_startups']))

    # intersetion of SpaCy, Stanford, NLTK
    intersection = list(set(intersection1) & set(nltk.iloc[i]['potential_startups']))
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
data.to_csv('combined_ner_output_org.csv', encoding='utf-8-sig')