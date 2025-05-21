import os 
import pandas as pd
from sklearn.feature_extraction.text import   CountVectorizer


def create_dataframe():
    data = {
        "id":[1,2,3,4,5,6,7,8,9,10],
        "review" :[
            "great food and ambiance  V2",
            "Terrible services",
            "amazing exprience",
            "food was  cold",
            "Loved the desserts",
            "not wirth the money",
            "excellent customer services",
            "the place was too crowded",
            "Best restaurant in the town",
            "average exprience in the town"

        ]
    }

    df = pd.DataFrame(data)

    return df
def saved_dataframe(df):
    if not os.path.exists("data"):
        os.makedirs("data")
    df.to_csv("data/data.csv",index=False)
    print("data is saved  in the data folder")

def process_data(k):
    df = pd.read_csv("data/data.csv")

    vecotorizer = CountVectorizer(max_features=k)
    vecotorizerised_data = vecotorizer.fit_transform(df['review'])
    features_name = vecotorizer.get_feature_names_out()



    # creating the new data framewith k new coloumns
    vectorized_df = pd.DataFrame(vecotorizerised_data.toarray(),columns=features_name)
    procced_df = pd.concat([df,vectorized_df],axis=1)

    return procced_df

if __name__== "__main__":
    df = create_dataframe()

    saved_dataframe(df)


    k = 5
    proced_df = process_data(k)
