import pandas as pd
import pathlib
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from InstructorEmbedding import INSTRUCTOR
import itertools
import warnings

warnings.filterwarnings("ignore")


def cleaner() -> None:
    """The function cleans the data, write the cleaned data to persistent disk and return none"""

    df = pd.read_csv("Train.csv")

    #filter out the selected features
    selected_df = df[['id', 'customer_age', 'job_type', 'housing_loan', 'personal_loan', 'education', 'balance']]

    #clean the id column
    selected_df["id"] = selected_df.loc[:,"id"].str.split("_").str[1].astype(int)
    selected_df = selected_df.set_index('id')

    #drop all rows that contains missing values
    selected_df.dropna(axis=0, inplace=True)

    #write the data set to disk
    data = pathlib.Path('/data')
    data.mkdir(parents=True, exist_ok=True)
    data_path = data / 'clean_data.csv'

    selected_df.to_csv(data_path, index=False)
    print(f"Saved the clean data at: {data_path}")

    return None



def embedding() -> None:
    """The function embedds the clean data in chunks of 5000, concatenate it write the result to a disk"""
    instruct_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", model_kwargs={"device": "cuda"})
    
    #Load the data
    loader = CSVLoader(file_path="/data/clean_data.csv")
    clean_data = loader.load()

    data_length = len(clean_data)
    data_gen = itertools.chain(clean_data)

    embeddings = []
    for _ in range(data_length):
        embeddings.append(instruct_embeddings.embed_query(next(iter(data_gen)).page_content))

    #convert it to a dataframe
    df_embeddings = pd.DataFrame(embeddings)

    #write the embeddings to file
    embedding_path = "df_embeddings.csv"
    df_embeddings.to_csv(embedding_path, index=False)
    print(f"Embedding dataframe saved at: {embedding_path}")

    return None
    

if __name__ == "__main__":
    cleaner()




