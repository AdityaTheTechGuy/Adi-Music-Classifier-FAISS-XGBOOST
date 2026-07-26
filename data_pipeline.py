import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# MACRO GENRE MAPPING
# ============================================================

GENRE_MAP = {

    "Rock": [
        "rock",
        "alt-rock",
        "alternative",
        "hard-rock",
        "punk",
        "punk-rock",
        "grunge",
        "psych-rock",
        "rock-n-roll",
        "rockabilly",
        "emo",
        "goth",
        "garage",
        "indie",
        "j-rock",
        "hardcore",
        "guitar"
    ],

    "Metal": [
        "metal",
        "heavy-metal",
        "black-metal",
        "death-metal",
        "metalcore",
        "grindcore"
    ],

    "Electronic": [
        "electronic",
        "edm",
        "electro",
        "house",
        "deep-house",
        "chicago-house",
        "techno",
        "minimal-techno",
        "trance",
        "dubstep",
        "drum-and-bass",
        "breakbeat",
        "ambient",
        "club",
        "dance",
        "disco",
        "hardstyle",

        # Previously unmapped
        "detroit-techno",
        "progressive-house",
        "idm",
        "industrial",
        "trip-hop"
    ],

    "Pop": [
        "pop",
        "indie-pop",
        "power-pop",
        "synth-pop",
        "k-pop",
        "j-pop",
        "j-idol",
        "cantopop",
        "mandopop",
        "british",

        # Previously unmapped
        "pop-film",
        "j-dance"
    ],

    "Hip-Hop": [
        "hip-hop"
    ],

    "R&B/Soul": [
        "r-n-b",
        "soul",
        "funk"
    ],

    "Folk": [
        "folk",
        "singer-songwriter",
        "bluegrass"
    ],

    "Country": [
        "country",
        "honky-tonk"
    ],

    "Jazz/Blues": [
        "jazz",
        "blues"
    ],

    "Classical": [
        "classical",
        "opera"
    ],

    "Reggae": [
        "reggae",
        "reggaeton",
        "dancehall",
        "dub",
        "ska"
    ],

    "Latin": [
        "latin",
        "latino",
        "salsa",
        "samba",
        "tango",
        "brazilian",

        # Previously unmapped
        "brazil",
        "forro",
        "mpb",
        "pagode",
        "sertanejo"
    ],

    "World": [
        "afrobeat",
        "indian",
        "iranian",
        "turkish",
        "malay",
        "world-music",

        # Regional/language categories
        "french",
        "german",
        "spanish",
        "swedish"
    ],

    # Acoustic / relaxed styles that don't fit cleanly
    # into Rock, Pop, Electronic, etc.
    "Easy Listening": [
        "acoustic",
        "chill",
        "new-age",
        "piano"
    ],

    # Children's and theatrical music
    "Children/Stage": [
        "children",
        "kids",
        "disney",
        "show-tunes"
    ],

    "Gospel": [
        "gospel"
    ]
}


# ============================================================
# GENRE -> MACRO GENRE
# ============================================================

def get_macro_genre(genre):
    """
    Convert the original Spotify track_genre into one of our
    broader macro genres.

    Genres that cannot be reasonably classified are kept as
    'Other' instead of being forced into an incorrect category.
    """

    genre = str(genre).strip().lower()

    for macro_genre, genres in GENRE_MAP.items():
        if genre in genres:
            return macro_genre

    return "Other"


# ============================================================
# LOAD AND CLEAN BASE DATA
# ============================================================

def load_and_clean_base_data(csv_path):
    """
    Load the Spotify dataset, normalize track/artist names,
    remove duplicate tracks and create macro genre labels.
    """

    df = pd.read_csv(csv_path)

    # --------------------------------------------------------
    # Normalize track names
    # --------------------------------------------------------

    df["track_name_norm"] = (
        df["track_name"]
        .str.replace(
            r"\s*[\(\[][^\]\)]*[\)\]]",
            "",
            regex=True
        )
        .str.strip()
        .str.lower()
    )

    # --------------------------------------------------------
    # Normalize artist names
    # --------------------------------------------------------

    df["artists_norm"] = (
        df["artists"]
        .str.lower()
        .str.strip()
    )

    # --------------------------------------------------------
    # Remove duplicate track + artist combinations
    # --------------------------------------------------------

    df = df.drop_duplicates(
        subset=[
            "track_name_norm",
            "artists_norm"
        ],
        keep="first"
    )

    # --------------------------------------------------------
    # Map Spotify genres -> macro genres
    # --------------------------------------------------------

    df["macro_genre"] = (
        df["track_genre"]
        .apply(get_macro_genre)
    )

    return df


# ============================================================
# FEATURE PREPROCESSING
# ============================================================

def build_preprocessing_pipeline():
    """
    Build preprocessing for the eight acoustic features used
    by FAISS and XGBoost.

    Tempo and loudness need scaling because their numerical
    ranges differ substantially from the other audio features.
    """

    scale_features = [
        "tempo",
        "loudness"
    ]

    passthrough_features = [
        "acousticness",
        "danceability",
        "energy",
        "instrumentalness",
        "speechiness",
        "valence"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "scaler",
                MinMaxScaler(),
                scale_features
            ),

            (
                "passthrough",
                "passthrough",
                passthrough_features
            )
        ],

        remainder="drop"
    )

    return preprocessor


# ============================================================
# FULL DATA PIPELINE
# ============================================================

def run_data_pipeline(csv_path="dataset.csv"):
    """
    Complete preprocessing pipeline.

    Returns:

    df
        Cleaned metadata containing macro_genre.

    df_cleaned
        Eight numerical features used by FAISS/XGBoost.

    all_engineered_features
        Names of the eight features.
    """

    # --------------------------------------------------------
    # 1. Load + clean data
    # --------------------------------------------------------

    df = load_and_clean_base_data(csv_path)


    # ========================================================
    # DIAGNOSTIC: GENRE MAPPING
    # ========================================================

    # Show Spotify genres that still fall into Other.

    unmapped = sorted(
        df.loc[
            df["macro_genre"] == "Other",
            "track_genre"
        ]
        .dropna()
        .unique()
    )

    print("\n========================================")
    print("UNMAPPED GENRES")
    print("========================================")

    if len(unmapped) == 0:
        print("None")

    else:
        for genre in unmapped:
            print(genre)


    # --------------------------------------------------------
    # Macro genre counts
    # --------------------------------------------------------

    print("\n========================================")
    print("MACRO GENRE DISTRIBUTION")
    print("========================================")

    print(
        df["macro_genre"]
        .value_counts()
    )


    # --------------------------------------------------------
    # Macro genre percentages
    # --------------------------------------------------------

    print("\n========================================")
    print("MACRO GENRE PERCENTAGES")
    print("========================================")

    percentages = (
        df["macro_genre"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print(percentages)


    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    print("\n========================================")
    print("DATASET INFORMATION")
    print("========================================")

    print(f"Tracks after deduplication: {len(df):,}")
    print(
        f"Number of macro genres: "
        f"{df['macro_genre'].nunique()}"
    )


    # ========================================================
    # 2. FEATURE PREPROCESSING
    # ========================================================

    preprocessor = build_preprocessing_pipeline()

    X_hybrid = preprocessor.fit_transform(df)


    # ColumnTransformer outputs scaled features first,
    # followed by passthrough features.

    scale_features = [
        "tempo",
        "loudness"
    ]

    passthrough_features = [
        "acousticness",
        "danceability",
        "energy",
        "instrumentalness",
        "speechiness",
        "valence"
    ]

    all_engineered_features = (
        scale_features
        + passthrough_features
    )


    # ========================================================
    # 3. CONVERT TO DATAFRAME
    # ========================================================

    df_cleaned = pd.DataFrame(
        X_hybrid,
        columns=all_engineered_features,
        index=df.index
    )


    # ========================================================
    # 4. RETURN RESULTS
    # ========================================================

    return (
        df,
        df_cleaned,
        all_engineered_features
    )


# ============================================================
# RUN THIS FILE DIRECTLY FOR TESTING
# ============================================================

if __name__ == "__main__":

    df, df_cleaned, features = run_data_pipeline(
        "dataset.csv"
    )

    print("\n========================================")
    print("PIPELINE COMPLETE")
    print("========================================")

    print("\nFeatures used:")

    for feature in features:
        print(f" - {feature}")

    print(
        f"\nFeature matrix shape: "
        f"{df_cleaned.shape}"
    )