import pandas as pd
from pathlib import Path

#harusnya lokasi nde kene

dataset_path = Path(
    "data/raw/behavioral/xAPI-Edu-Data.csv"
)

# pastikno file enek

if not dataset_path.exists():
    raise FileNotFoundError(f"File {dataset_path} " 
            )

# baca dataset 

df = pd.read_csv(dataset_path) 


#perform data yok
print ("\nFirst 5 Rows:")
print(df.head())

#size data
print("\nDataset size:")
print(df.shape)

#nama kolom
print("\nColumn Name:")
print(df.columns.tolist())

#info tipe data
print("\nData Types:")
print(df.dtypes)

#missing data
print("\nMissing Data:")
print(df.isnull().sum())

#duplicate
print("\nDupicate data:")
print(df.duplicated().sum())

#distribusi target
print("\nTarget Distribution:")
print(df['Class'].value_counts())

print("\n===== INCEPTION COMPLETED =====")


print("\n==== DUPLICATE ANALYSIS ====")

duplicates = df[df.duplicated(keep=False)]

print("Total duplicate rows:", len(duplicates))
print(duplicates)

# statistik fitur numerik
print("\n==== NUMERICAL FEATURE STATISTICS =====")

numerical_columns = [
    "raisedhands",
    "VisITedResources",
    "AnnouncementsView",
    "Discussion"
]

print(df[numerical_columns].describe())

print("\n==== CATEGORICAL ANALYSIS =====")

categorical_columns = [
   "gender",
    "NationalITy",
    "PlaceofBirth",
    "StageID",
    "GradeID",
    "SectionID",
    "Topic",
    "Semester",
    "Relation",
    "ParentAnsweringSurvey",
    "ParentschoolSatisfaction",
    "StudentAbsenceDays"
]

for column in categorical_columns:

    print(f"\n{column}:")
    print(df[column].value_counts())

    #target precentage

    print("\n==== TARGET PRECENTAGE ====")

    target_percentage = (
        df["Class"].value_counts(normalize=True) * 100
    )

    print(target_percentage.round(2))

    #Distribusi data target

    print("\n==== BEHAVIOR BY CLASS ====")


    behavior_analysis = df.groupby("Class")[
        numerical_columns
    ].mean()

    print(behavior_analysis.round(2))


    print("\n==== EXPLORATION COMPLETED ====")