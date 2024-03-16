import numpy as np
import pandas as pd
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

DATA_PATH = "data"
OUTPUT_PATH = "output"
SCORING = "neg_log_loss"


def get_team_features() -> pd.DataFrame:
    detailed_results_season = extract_data("RegularSeasonDetailedResults")
    detailed_results_tournament = extract_data("NCAATourneyDetailedResults")
    # Attribute tournament results for year N to season N+1.
    detailed_results_tournament["Season"] += 1
    seeds = extract_data("NCAATourneySeeds")
    rankings = extract_data("MasseyOrdinals")
    team_features_season = detailed_results_to_team_features(detailed_results_season)
    team_features_tournament = detailed_results_to_team_features(
        detailed_results_tournament
    )
    team_features_seed = seeds_to_team_features(seeds)
    team_features_ranking = rankings_to_team_features(rankings)
    team_features = merge_features(
        team_features_season,
        team_features_tournament,
        team_features_seed,
        team_features_ranking,
    )
    return team_features


def extract_data(filename: str, data_path: str = DATA_PATH) -> pd.DataFrame:
    mens_filepath = f"{data_path}/M{filename}.csv"
    womens_filepath = f"{data_path}/W{filename}.csv"
    try:
        df_mens = pd.read_csv(mens_filepath)
    except FileNotFoundError:
        df_mens = None
    try:
        df_womens = pd.read_csv(womens_filepath)
    except FileNotFoundError:
        df_womens = None
    df = pd.concat([df_mens, df_womens])
    return df


def detailed_results_to_team_features(detailed_results: pd.DataFrame) -> pd.DataFrame:
    df = detailed_results.copy()
    df = clean_detailed_results(df)
    df = transform_game_to_team(df)
    df = transform_team_results(df)
    return df


def seeds_to_team_features(seeds: pd.DataFrame) -> pd.DataFrame:
    df = seeds.copy()
    mask = df["Season"] > 2002
    df = df[mask]
    df["Seed"] = df["Seed"].str.replace(r"\D+", "", regex=True)
    df["Seed"] = df["Seed"].astype(int)
    return df


def rankings_to_team_features(rankings: pd.DataFrame) -> pd.DataFrame:
    df = rankings.copy()
    mask = df["RankingDayNum"] == df["RankingDayNum"].max()
    df = df[mask]
    df.drop(["SystemName", "RankingDayNum"], axis=1, inplace=True)
    df = df.groupby(["Season", "TeamID"]).agg("median")
    df = df.reset_index()
    return df


def get_game_outcomes(df):
    input_rows = df.to_records()
    output_rows = []
    for input_row in input_rows:
        output_rows.extend(parse_row(input_row))
    out_df = pd.DataFrame(output_rows)
    return out_df


def parse_row(row):
    season = row["Season"]
    winning_team_id = row["WTeamID"]
    losing_team_id = row["LTeamID"]
    if winning_team_id < losing_team_id:
        small_id = winning_team_id
        big_id = losing_team_id
        outcome = True
    elif losing_team_id < winning_team_id:
        small_id = losing_team_id
        big_id = winning_team_id
        outcome = False
    records = [
        {
            "ID": f"{season}_{small_id}_{big_id}",
            "Season": season,
            "LowID": small_id,
            "HighID": big_id,
            "Win": outcome,
        },
        {
            "ID": f"{season}_{big_id}_{small_id}",
            "Season": season,
            "LowID": big_id,
            "HighID": small_id,
            "Win": not outcome,
        },
    ]
    return records


def clean_detailed_results(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(["WLoc", "DayNum"], axis=1)


def transform_game_to_team(game_results: pd.DataFrame) -> pd.DataFrame:
    winners = rename_columns(game_results, "W")
    loosers = rename_columns(game_results, "L")
    team_results = pd.concat((winners, loosers))
    team_results.drop(["TeamIDOpp"], axis=1, inplace=True)
    return team_results


def transform_team_results(df: pd.DataFrame) -> pd.DataFrame:
    df = df.groupby(["Season", "TeamID"]).median()
    df["FGP"] = df["FGM"] / df["FGA"]
    df["FGP3"] = df["FGM3"] / df["FGA3"]
    df["FTP"] = df["FTM"] / df["FTA"]
    df["FGPOpp"] = df["FGMOpp"] / df["FGAOpp"]
    df["FGP3Opp"] = df["FGM3Opp"] / df["FGA3Opp"]
    df["FTPOpp"] = df["FTMOpp"] / df["FTAOpp"]
    return df.reset_index()


def rename_columns(df: pd.DataFrame, team_prefix: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = (rename_column(column_name, team_prefix) for column_name in df.columns)
    return df


def rename_column(column_name: str, team_prefix: str) -> pd.DataFrame:
    if team_prefix == "W":
        opponent_prefix = "L"
    elif team_prefix == "L":
        opponent_prefix = "W"
    else:
        raise ValueError
    if column_name.startswith(team_prefix):
        column_name = column_name.lstrip(team_prefix)
    elif column_name.startswith(opponent_prefix):
        column_name = f"{column_name.lstrip(opponent_prefix)}Opp"
    return column_name


def merge_features(
    season_features: pd.DataFrame,
    tournament_features: pd.DataFrame,
    seed_features: pd.DataFrame,
    ranking_features: pd.DataFrame,
) -> pd.DataFrame:
    features = pd.merge(
        season_features,
        tournament_features,
        how="inner",
        on=["Season", "TeamID"],
        suffixes=("Reg", "Tou"),
    )
    features = features.merge(seed_features, how="inner", on=["Season", "TeamID"])
    features = features.merge(
        ranking_features,
        how="left",
        on=["Season", "TeamID"],
    )
    return features


def merge_outcomes_with_features(
    outcomes: pd.DataFrame, features: pd.DataFrame, how: str = "inner"
) -> pd.DataFrame:
    feature_names = [
        name for name in features.columns if name not in ["Season", "TeamID", "Gender"]
    ]
    data = pd.merge(
        outcomes,
        features,
        how=how,
        left_on=["Season", "HighID"],
        right_on=["Season", "TeamID"],
    )
    data = pd.merge(
        data,
        features,
        how=how,
        left_on=["Season", "LowID"],
        right_on=["Season", "TeamID"],
        suffixes=("High", "Low"),
    )
    for name in feature_names:
        data[f"{name}Diff"] = data[f"{name}High"] - data[f"{name}Low"]
        data.drop([f"{name}High", f"{name}Low"], axis=1, inplace=True)
    data.drop(
        ["Season", "HighID", "LowID", "TeamIDHigh", "TeamIDLow"], axis=1, inplace=True
    )
    data.set_index("ID", inplace=True)
    return data


def get_submission_outcomes() -> pd.DataFrame:
    sample_submission = pd.read_csv(
        f"./kaggle/input/{COMPETITION_NAME}/SampleSubmission2023.csv"
    )
    df = sample_submission.copy()
    df.drop("Pred", axis=1, inplace=True)
    df[["Season", "LowID", "HighID"]] = df["ID"].str.split("_", expand=True)
    df[["Season", "LowID", "HighID"]] = df[["Season", "LowID", "HighID"]].astype(int)
    return df


def objective(trial: optuna.Trial, X_train, y_train, scoring=SCORING):
    params = {
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }

    if params["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        params["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
        # minimum child weight, larger the term more conservative the tree.
        params["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        params["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        params["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        params["grow_policy"] = trial.suggest_categorical(
            "grow_policy", ["depthwise", "lossguide"]
        )

    if params["booster"] == "dart":
        params["sample_type"] = trial.suggest_categorical(
            "sample_type", ["uniform", "weighted"]
        )
        params["normalize_type"] = trial.suggest_categorical(
            "normalize_type", ["tree", "forest"]
        )
        params["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        params["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    model = train_model(params, X_train, y_train, scoring)
    score = np.mean(cross_val_score(model, X_train, y_train, scoring=scoring, cv=5))
    return score


def run_study(X_train, y_train):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=100)
    return study


def train_model(params, X, y, scoring):
    model = XGBClassifier(
        objective="binary:logistic",
        tree_method="exact",
        verbosity=0,
        boosting_type="gbdt",
        **params,
    )
    model = model.fit(X, y)
    return model
