import pandas as pd
from pymongo import MongoClient
from openai import OpenAI
from config import mongo_uri, database_name, collection_name, openai_api_key

client = OpenAI(
    api_key=openai_api_key,  # this is also the default, it can be omitted
)


# Function to load the first 100 rows from CSV and filter columns
def load_and_filter_csv(file_path, accepted_columns, nrows=500):
    # Load the first 100 rows from the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path, usecols=accepted_columns, nrows=nrows)

    return df


# Function to load column descriptions from a separate CSV
def load_column_descriptions(description_file):
    # Load the CSV containing column name and description mapping
    descriptions_df = pd.read_csv(description_file)

    # Convert the mapping to a dictionary {column_name: description}
    description_mapping = dict(zip(descriptions_df['VARIABLE NAME'], descriptions_df['DESCRIPTION']))

    return description_mapping


# Function to create a text column by joining column descriptions and row values into a natural language string
def create_text_column(df, description_mapping, separator=" | ", empty_value="Unknown"):
    # Combine all columns into a 'text_column' where each entry is structured with descriptions and values
    # Adding a separator between columns for better readability and handling NaN values
    df['text_column'] = df.apply(lambda row: separator.join(
        [f"{description_mapping.get(col, col)}: {row[col] if pd.notna(row[col]) else empty_value}" for col in
         df.columns]), axis=1)

    return df


# Function to get the embedding for a piece of text using the OpenAI client
def get_embedding(text, model="text-embedding-ada-002"):
    # Clean the text (replace newlines with spaces)
    text = text.replace("\n", " ")

    # Call the OpenAI API to get the embedding
    response = client.embeddings.create(input=[text], model=model)

    # Return the embedding vector from the response
    return response.data[0].embedding


# Function to embed the 'text_column' in the DataFrame
def embed_text(df):
    # Apply the get_embedding function to each row's 'text_column'
    df['embedding'] = df['text_column'].apply(lambda text: get_embedding(text))
    return df


# Function to insert data into MongoDB
def insert_into_mongodb(df):
    # Convert DataFrame to dictionary records format
    records = df.to_dict(orient='records')

    # Connect to MongoDB
    client = MongoClient(mongo_uri)
    db = client[database_name]
    collection = db[collection_name]

    # Insert records into MongoDB
    collection.insert_many(records)


# Function to create a vector index on the 'embedding' field in MongoDB
def create_vector_index():
    # Connect to MongoDB
    client = MongoClient(mongo_uri)
    db = client[database_name]
    collection = db[collection_name]

    # Create a vector index on the 'embedding' field
    collection.create_index(
        [
            ("embedding", {
                "vector": {
                    "type": "float32",  # Type of the vector (OpenAI embeddings are float32)
                    "dimensions": 1536  # Number of dimensions in OpenAI Ada embeddings
                }
            })
        ],
        name="embedding_vector_index"
    )
    print("Vector index created on 'embedding' field.")


# Main function to run the process
def main(data_file, description_file, accepted_columns):
    # Load and filter the first 100 rows from the main CSV
    df = load_and_filter_csv(data_file, accepted_columns)

    # Load the column descriptions from a separate CSV
    description_mapping = load_column_descriptions(description_file)

    # Create a natural language text column using the descriptions
    df = create_text_column(df, description_mapping)

    # Generate embeddings using OpenAI API
    df = embed_text(df)

    # Insert data into MongoDB
    insert_into_mongodb(df)

    # Create a vector index on the 'embedding' field
    create_vector_index()

    print(f"Data inserted into MongoDB collection '{collection_name}' and vector index created.")


if __name__ == "__main__":
    # Example CSV file paths and list of accepted columns
    data_file_path = './files/KIDPAN_DATA_clean.csv'  # Replace with the path to your main data CSV
    description_file_path = './files/kinpan_data_dictionary.csv'  # Replace with the path to your description CSV

    # Example list of accepted columns (without renaming)
    accepted_columns = ['ADMISSION_DATE', 'ADMIT_DATE_DON', 'AGE', 'AGE_DIAB', 'AGE_DON', 'AGE_GROUP', 'BMI_CALC',
                        'BMI_DON_CALC', 'BMI_TCR', 'CANCER_FREE_INT_DON', 'CANCER_SITE_DON', 'CARDARREST_NEURO',
                        'CITIZEN_COUNTRY', 'CITIZEN_COUNTRY_DON', 'CITIZENSHIP', 'CITIZENSHIP_DON', 'CLIN_INFECT_DON',
                        'COD_CAD_DON', 'COD_KI', 'COD_PA', 'COD_WL', 'COD2_KI', 'COD2_PA', 'COD3_KI', 'COD3_PA',
                        'COMPL_ABSC', 'COMPL_ANASLK', 'COMPL_PANCREA', 'COMPOSITE_DEATH_DATE',
                        'CONTIN_ALCOHOL_OLD_DON', 'CONTIN_CIG_DON', 'CONTIN_COCAINE_DON', 'CONTIN_IV_DRUG_OLD_DON',
                        'CONTIN_OTH_DRUG_DON', 'DATA_TRANSPLANT', 'DATA_WAITLIST', 'DAYSWAIT_ALLOC', 'DAYSWAIT_CHRON',
                        'DAYSWAIT_CHRON_KI', 'DAYSWAIT_CHRON_PA', 'DEATH_CIRCUM_DON', 'DEATH_DATE', 'DEATH_MECH_DON',
                        'DGN_TCR', 'DGN2_TCR', 'DIAG_KI', 'DIAG_PA', 'DIAL_DATE', 'DIAL_TRR', 'DISCHARGE_DATE',
                        'DISTANCE', 'DONOR_ID', 'EDUCATION', 'EDUCATION_DON', 'END_BMI_CALC',
                        'END_DATE', 'ETHCAT', 'ETHCAT_DON', 'ETHNICITY', 'FAILDATE_KI', 'FAILDATE_PA', 'GENDER',
                        'GENDER_DON', 'GFR', 'GFR_DATE', 'GTIME_KI', 'GTIME_PA', 'GSTATUS_KI',
                        'GSTATUS_PA', 'INIT_AGE', 'INIT_BMI_CALC', 'INIT_DATE', 'INIT_HGT_CM', 'INIT_STAT',
                        'INIT_WGT_KG', 'LIV_DON_TY', 'ORGAN', 'REM_CD', 'TRANSPLANT_TIME', 'TRANSPLANTTIMEZONEID',
                        'TRTREJ1Y_KI', 'TRTREJ1Y_PA', 'TRTREJ6M_KI', 'TRTREJ6M_PA', 'TX_DATE', 'TX_PROCEDUR_TY_KI',
                        'TX_PROCEDUR_TY_PA', 'TX_TYPE', 'WGT_KG_CALC', 'WGT_KG_DON_CALC', 'WGT_KG_TCR',
                        'WL_ORG', 'WT_QUAL_DATE', 'WORK_INCOME_TCR',
                        'WORK_INCOME_TRR']  # Replace with your actual accepted columns

    # Run the process
    main(data_file_path, description_file_path, accepted_columns)
