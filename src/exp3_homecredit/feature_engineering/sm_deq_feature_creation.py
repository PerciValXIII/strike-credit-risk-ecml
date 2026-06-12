import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, List


class BureauFeatureEngineer:
    def __init__(self, application_path: str, bureau_path: str, bureau_balance_path: str):
        self.application_path = application_path
        self.bureau_path = bureau_path
        self.bureau_balance_path = bureau_balance_path

        self.df_application = None
        self.df_bureau = None
        self.df_bureau_balance = None
        self.features = None

    def load_data(self):
        self.df_application = pd.read_csv(self.application_path)
        self.df_bureau = pd.read_csv(self.bureau_path)
        self.df_bureau_balance = pd.read_csv(self.bureau_balance_path)

    def add_custom_features(self):
        df = self.df_bureau.copy()

        df["ENDDATE_DIFF"] = df["DAYS_ENDDATE_FACT"] - df["DAYS_CREDIT_ENDDATE"]
        df["DAYS_CREDIT_PLAN"] = df["DAYS_CREDIT_ENDDATE"] - df["DAYS_CREDIT"]

        credit_mean_by_type = df.groupby("CREDIT_TYPE")["AMT_CREDIT_SUM"].mean().to_dict()
        df["MEAN_AMT_CREDIT_BY_CREDIT_TYPE"] = df["CREDIT_TYPE"].map(credit_mean_by_type)

        df["AMT_CREDIT_SUM_RATIO"] = df["AMT_CREDIT_SUM"] / df["MEAN_AMT_CREDIT_BY_CREDIT_TYPE"]
        df["AMT_CREDIT_DEBT_PERC"] = df["AMT_CREDIT_SUM_DEBT"] / df["AMT_CREDIT_SUM"]
        df["AMT_CREDIT_DEBT_DIFF"] = df["AMT_CREDIT_SUM_DEBT"] - df["AMT_CREDIT_SUM"]
        df["AMT_CREDIT_ANNUITY_PERC"] = df["AMT_ANNUITY"] / df["AMT_CREDIT_SUM"]

        self.df_bureau = df

    def perform_aggregation(self, df: pd.DataFrame, agg_funcs: Dict[str, List[str]], prefix: str):
        agg_dict = {}
        for col, funcs in agg_funcs.items():
            agg_dict[col] = funcs

        grouped = df.groupby("SK_ID_CURR").agg(agg_dict)
        # Flatten MultiIndex columns
        grouped.columns = [f"{prefix}{col}_{stat}" for col, stat in grouped.columns]
        grouped.reset_index(inplace=True)
        return grouped

    def engineer_features(self):
        self.add_custom_features()

        # Base Aggregations
        agg_funcs = {
            'DAYS_CREDIT': ['count', 'mean'],
            'CREDIT_DAY_OVERDUE': ['mean', 'sum'],
            'DAYS_CREDIT_ENDDATE': ['mean'],
            'DAYS_ENDDATE_FACT': ['mean'],
            'AMT_CREDIT_MAX_OVERDUE': ['mean', 'sum'],
            'CNT_CREDIT_PROLONG': ['mean', 'sum'],
            'AMT_CREDIT_SUM': ['mean', 'sum'],
            'AMT_CREDIT_SUM_DEBT': ['mean', 'sum'],
            'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
            'AMT_CREDIT_SUM_OVERDUE': ['mean', 'sum'],
            'CREDIT_TYPE': ['nunique'],
            'DAYS_CREDIT_UPDATE': ['mean'],
            'AMT_ANNUITY': ['mean', 'sum'],
            'ENDDATE_DIFF': ['mean'],
            'AMT_CREDIT_SUM_RATIO': ['mean', 'max'],
            'DAYS_CREDIT_PLAN': ['mean', 'sum'],
            'AMT_CREDIT_DEBT_PERC': ['mean', 'min', 'max'],
            'AMT_CREDIT_DEBT_DIFF': ['mean', 'sum']
        }

        base_agg = self.perform_aggregation(self.df_bureau, agg_funcs, "b_")

        # Filtered aggregations
        filters = [
            (self.df_bureau["CREDIT_ACTIVE"] == "Active", "b_active_"),
            (self.df_bureau["CREDIT_ACTIVE"] == "Closed", "b_closed_"),
            (self.df_bureau["CREDIT_TYPE"] == "Consumer credit", "b_consumer_"),
            (self.df_bureau["CREDIT_TYPE"] == "Credit card", "b_credit_"),
            (self.df_bureau["CREDIT_TYPE"] == "Car loan", "b_car_"),
            (self.df_bureau["CREDIT_TYPE"] == "Mortgage", "b_mortgage_"),
            (self.df_bureau["CREDIT_TYPE"] == "Microloan", "b_micro_"),
            (self.df_bureau["DAYS_CREDIT"] >= -720, "b_720_"),
            (self.df_bureau["DAYS_CREDIT"] >= -365, "b_365_"),
        ]

        for condition, prefix in filters:
            filtered_df = self.df_bureau[condition].copy()
            filtered_agg = self.perform_aggregation(filtered_df, agg_funcs, prefix)
            base_agg = base_agg.merge(filtered_agg, on="SK_ID_CURR", how="left")
        
        # Join with TARGET from application data
        df_target = self.df_application[["SK_ID_CURR", "TARGET"]]
        self.features = base_agg.merge(df_target, on="SK_ID_CURR", how="inner")

    def get_features(self) -> pd.DataFrame:
        if self.features is None:
            raise ValueError("Features have not been engineered yet. Call `engineer_features()` first.")
        return self.features


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parents[1]
    
    engineer = BureauFeatureEngineer(
        application_path=os.path.join(root_dir, "data", "application_train.csv"),
        bureau_path=os.path.join(root_dir, "data", "bureau.csv"),
        bureau_balance_path=os.path.join(root_dir, "data", "bureau_balance.csv")
    )
    engineer.load_data()
    engineer.engineer_features()
    features_df = engineer.get_features()

    # save in data
    output_path = os.path.join(root_dir, "data", "deq_features_level1.csv")
    features_df.to_csv(output_path, index=False)

