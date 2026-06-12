import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import mode
from pathlib import Path
import os

def create_vintage_features(app_train: pd.DataFrame, prev_app: pd.DataFrame, bureau_df: pd.DataFrame) -> pd.DataFrame:

    previous_application = prev_app
    application_train = app_train
    bureau = bureau_df

    time_periods = [3, 6, 12, 24, 36]
    aggregators = ['sum', 'mean', 'max', 'min', 'std']

    def vin_generic_aggregator(df, groupby_column, agg_column, agg_funcs):
        aggregated_df = df.groupby(groupby_column).agg({agg_column: agg_funcs}).reset_index()
        aggregated_df.columns = ['_'.join(col).rstrip('_') for col in aggregated_df.columns.values]
        return aggregated_df

    # 1. Vintage - Months Since Last Approved Loan (based on DAYS_DECISION)
    def vin_months_since_last_approved(previous_application, agg_funcs=['max']):
        last_approved = previous_application[previous_application['NAME_CONTRACT_STATUS'] == 'Approved']
        last_approved = vin_generic_aggregator(last_approved, 'SK_ID_CURR', 'DAYS_DECISION', agg_funcs)
        for agg_func in agg_funcs:
            last_approved[f'vin_months_since_last_approved_{agg_func}'] = last_approved[f'DAYS_DECISION_{agg_func}'] / -30
        return last_approved[['SK_ID_CURR'] + [f'vin_months_since_last_approved_{agg_func}' for agg_func in agg_funcs]]
    vin_months_since_last_approved(previous_application)

    # 2. Vintage - Days Since Last Loan Rejection (based on DAYS_DECISION)
    def vin_days_since_last_rejection(previous_application, agg_funcs=['max']):
        last_rejected = previous_application[previous_application['NAME_CONTRACT_STATUS'] == 'Refused']
        last_rejected = vin_generic_aggregator(last_rejected, 'SK_ID_CURR', 'DAYS_DECISION', agg_funcs)
        for agg_func in agg_funcs:
            last_rejected[f'days_since_last_rejection_{agg_func}'] = last_rejected[f'DAYS_DECISION_{agg_func}'] * -1
        return last_rejected[['SK_ID_CURR'] + [f'days_since_last_rejection_{agg_func}' for agg_func in agg_funcs]]

    vin_days_since_last_rejection(previous_application, aggregators)


    # 3. Vintage - Number of Loans Approved in the Last N Months (based on DAYS_DECISION)
    def vin_num_loans_approved_last_n_months(previous_application, months=12):
        recent_approved = previous_application[(previous_application['NAME_CONTRACT_STATUS'] == 'Approved') &
                                               (previous_application['DAYS_DECISION'] >= -30 * months)]
        num_approved = recent_approved.groupby('SK_ID_CURR').size().reset_index(name=f'vin_num_loans_approved_last_{months}_months')
        return num_approved

    vin_num_loans_approved_last_n_months(previous_application)

    # 4. Vintage - Days Since Last Payment Default (based on DAYS_CREDIT)
    def vin_days_since_last_default(bureau, agg_funcs=['min']):
        defaulted = bureau[bureau['CREDIT_DAY_OVERDUE'] > 0]
        last_default = vin_generic_aggregator(defaulted, 'SK_ID_CURR', 'DAYS_CREDIT', agg_funcs)
        for agg_func in agg_funcs:
            last_default[f'days_since_last_default_{agg_func}'] = last_default[f'DAYS_CREDIT_{agg_func}'] * -1
        return last_default[['SK_ID_CURR'] + [f'days_since_last_default_{agg_func}' for agg_func in agg_funcs]]

    vin_days_since_last_default(bureau)

    # 5. Vintage - Number of Previous Applications Over Time (based on DAYS_DECISION)
    def vin_num_previous_applications_last_n_months(previous_application, months=12):
        recent_apps = previous_application[previous_application['DAYS_DECISION'] >= -30 * months]
        num_apps = recent_apps.groupby('SK_ID_CURR').size().reset_index(name=f'num_previous_applications_last_{months}_months')
        return num_apps

    vin_num_previous_applications_last_n_months(previous_application)

    # 6. Vintage - Last Loan Taken by Category (based on DAYS_DECISION)
    def vin_last_loan_taken_by_category(previous_application, agg_funcs=['max']):
        last_loan = previous_application.groupby(['SK_ID_CURR', 'NAME_CONTRACT_TYPE'])['DAYS_DECISION'].agg(agg_funcs).reset_index()
        for agg_func in agg_funcs:
            last_loan[f'vin_days_since_last_loan_{agg_func}'] = last_loan[agg_func] * -1
        last_loan_pivot = last_loan.pivot(index='SK_ID_CURR', columns='NAME_CONTRACT_TYPE', values=[f'vin_days_since_last_loan_{agg_func}' for agg_func in agg_funcs]).reset_index()
        last_loan_pivot.columns = ['SK_ID_CURR'] + [f'vin_last_loan_{col[1].lower()}_{col[0].split("_")[-2]}' for col in last_loan_pivot.columns[1:]]
        return last_loan_pivot


    vin_last_loan_taken_by_category(previous_application)


    # 7. Vintage - Days Since Last Loan Closure (based on DAYS_ENDDATE_FACT)
    def vin_days_since_last_loan_closure(bureau, agg_funcs=['max']):
        closed_loans = bureau[bureau['CREDIT_ACTIVE'] == 'Closed']
        last_closure = vin_generic_aggregator(closed_loans, 'SK_ID_CURR', 'DAYS_ENDDATE_FACT', agg_funcs)
        for agg_func in agg_funcs:
            last_closure[f'vin_days_since_last_loan_closure_{agg_func}'] = last_closure[f'DAYS_ENDDATE_FACT_{agg_func}'] * -1
        return last_closure[['SK_ID_CURR'] + [f'vin_days_since_last_loan_closure_{agg_func}' for agg_func in agg_funcs]]

    vin_days_since_last_loan_closure(bureau, aggregators)


    # 8. Vintage - Days Since First Loan Taken (based on DAYS_CREDIT)
    def vin_days_since_first_loan_taken(bureau, agg_funcs=['max']):
        first_loan = vin_generic_aggregator(bureau, 'SK_ID_CURR', 'DAYS_CREDIT', agg_funcs)
        for agg_func in agg_funcs:
            first_loan[f'vin_days_since_first_loan_taken_{agg_func}'] = first_loan[f'DAYS_CREDIT_{agg_func}'] * -1
        return first_loan[['SK_ID_CURR'] + [f'vin_days_since_first_loan_taken_{agg_func}' for agg_func in agg_funcs]]
    vin_days_since_first_loan_taken(bureau, aggregators)

    # 9. Vintage - Number of Times Delinquent in Last N Months (based on DAYS_CREDIT)
    def vin_num_times_delinquent_last_n_months(bureau, months=6):
        recent_delinquencies = bureau[(bureau['CREDIT_DAY_OVERDUE'] > 0) &
                                      (bureau['DAYS_CREDIT'] >= -30 * months)]
        num_delinquent = recent_delinquencies.groupby('SK_ID_CURR').size().reset_index(name=f'vin_num_times_delinquent_last_{months}_months')
        return num_delinquent

    vin_num_times_delinquent_last_n_months(bureau)

    # 10. Vintage - Number of Overdue Loans in Last N Months (based on DAYS_CREDIT)
    def vin_num_overdue_loans_last_n_months(bureau, months=6):
        recent_overdue = bureau[(bureau['AMT_CREDIT_SUM_OVERDUE'] > 0) &
                                (bureau['DAYS_CREDIT'] >= -30 * months)]
        num_overdue = recent_overdue.groupby('SK_ID_CURR').size().reset_index(name=f'vin_num_overdue_loans_last_{months}_months')
        return num_overdue
    vin_num_overdue_loans_last_n_months(bureau)

    # 11. Vintage - Average Time Between Loans (based on DAYS_DECISION)
    def vin_avg_time_between_loans(previous_application):
        approved_loans = previous_application[previous_application['NAME_CONTRACT_STATUS'] == 'Approved']
        approved_loans = approved_loans.sort_values(by=['SK_ID_CURR', 'DAYS_DECISION'])
        approved_loans['time_diff'] = approved_loans.groupby('SK_ID_CURR')['DAYS_DECISION'].diff().abs()
        avg_time_between = approved_loans.groupby('SK_ID_CURR')['time_diff'].mean().reset_index(name='vin_avg_time_between_loans')
        return avg_time_between
    vin_avg_time_between_loans(previous_application)

    # 12. Vintage - Number of Active Loans (based on CREDIT_ACTIVE)
    def vin_num_active_loans(bureau):
        active_loans = bureau[bureau['CREDIT_ACTIVE'] == 'Active'].groupby('SK_ID_CURR').size().reset_index(name='vin_num_active_loans')
        return active_loans
    vin_num_active_loans(bureau)

    # 13. Vintage - Loan prolongation frequency
    def vin_loan_prolongation_frequency(bureau):
        prolongation_frequency = bureau.groupby('SK_ID_CURR')['CNT_CREDIT_PROLONG'].sum().reset_index(name='vin_loan_prolongation_frequency')
        return prolongation_frequency

    vin_loan_prolongation_frequency(bureau)

    # 14. Vintage - Total Overdue Amount in Last N Months (based on AMT_CREDIT_SUM_OVERDUE)
    def vin_total_overdue_amount_last_n_months(bureau, months=6):
        recent_overdue = bureau[(bureau['DAYS_CREDIT'] >= -30 * months)]
        total_overdue = recent_overdue.groupby('SK_ID_CURR')['AMT_CREDIT_SUM_OVERDUE'].sum().reset_index(name=f'vin_total_overdue_amount_last_{months}_months')
        return total_overdue
    vin_total_overdue_amount_last_n_months(bureau)

    # 15. Vintage - Recency of Credit Update (based on DAYS_CREDIT_UPDATE)
    def vin_days_since_credit_update(bureau, agg_funcs=['min']):
        credit_update = vin_generic_aggregator(bureau, 'SK_ID_CURR', 'DAYS_CREDIT_UPDATE', agg_funcs)
        for agg_func in agg_funcs:
            credit_update[f'vin_days_since_credit_update_{agg_func}'] = credit_update[f'DAYS_CREDIT_UPDATE_{agg_func}'] * -1
        return credit_update[['SK_ID_CURR'] + [f'vin_days_since_credit_update_{agg_func}' for agg_func in agg_funcs]]

    vin_days_since_credit_update(bureau)

    # 16. Vintage - Number of Times Credit Limit Was Reached (based on AMT_CREDIT_SUM and AMT_CREDIT_SUM_LIMIT)
    def vin_num_times_credit_limit_reached(bureau):
        limit_reached = bureau[bureau['AMT_CREDIT_SUM'] >= bureau['AMT_CREDIT_SUM_LIMIT']].groupby('SK_ID_CURR').size().reset_index(name='vin_num_times_credit_limit_reached')
        return limit_reached
    vin_num_times_credit_limit_reached(bureau)

    # 17. Vintage - Average Number of Rejections Over Time (based on DAYS_DECISION)
    def vin_avg_rejections_last_n_months(previous_application, months=12):
        results = []
        # for months in time_periods:
        rejections = previous_application[(previous_application['NAME_CONTRACT_STATUS'] == 'Refused') &
                                          (previous_application['DAYS_DECISION'] >= -30 * months)]
        avg_rejections = rejections.groupby('SK_ID_CURR').size().reset_index(name=f'vin_avg_rejections_last_{months}_months')
        results.append(avg_rejections)
        return pd.concat(results, axis=1)
    vin_avg_rejections_last_n_months(previous_application)

    # 18. Vintage - Days Since Last Document Update (based on DAYS_ID_PUBLISH)
    def vin_days_since_last_document_update(application_train, agg_funcs=['max']):
        last_doc_update = vin_generic_aggregator(application_train, 'SK_ID_CURR', 'DAYS_ID_PUBLISH', agg_funcs)
        for agg_func in agg_funcs:
            last_doc_update[f'vin_days_since_last_document_update_{agg_func}'] = last_doc_update[f'DAYS_ID_PUBLISH_{agg_func}'] * -1
        return last_doc_update[['SK_ID_CURR'] + [f'vin_days_since_last_document_update_{agg_func}' for agg_func in agg_funcs]]

    vin_days_since_last_document_update(application_train)

    # List of functions to calculate vintage variables
    vintage_functions = [
        vin_months_since_last_approved,
        vin_days_since_last_rejection,
        vin_days_since_last_default,
        vin_num_previous_applications_last_n_months,
        vin_last_loan_taken_by_category,
        vin_days_since_last_loan_closure,
        vin_days_since_first_loan_taken,
        vin_num_times_delinquent_last_n_months,
        vin_avg_time_between_loans,
        vin_loan_prolongation_frequency,
        vin_days_since_credit_update,
        vin_num_times_credit_limit_reached,
        vin_days_since_last_document_update
    ]

    # Create dictionaries to store feature DataFrames for each source separately
    application_features = {}
    previous_application_features = {}
    bureau_features = {}

    # Functions to generate time-period-based features
    for months in time_periods:
        time_based_functions = [
            vin_num_loans_approved_last_n_months,
            vin_num_previous_applications_last_n_months,
            vin_num_times_delinquent_last_n_months,
            vin_num_overdue_loans_last_n_months,
            vin_total_overdue_amount_last_n_months,
            vin_avg_rejections_last_n_months
        ]

        for func in time_based_functions:
            # Determine which data source to use
            if 'previous_application' in func.__code__.co_varnames:
                data_source = previous_application
                storage_dict = previous_application_features
            elif 'bureau' in func.__code__.co_varnames:
                data_source = bureau
                storage_dict = bureau_features
            else:
                data_source = application_train
                storage_dict = application_features

            # Generate feature DataFrame
            feature_df = func(data_source, months=months)

            # Store the generated DataFrame in the appropriate dictionary
            feature_name = f"{func.__name__}_{months}months"
            storage_dict[feature_name] = feature_df

    # Functions to generate non-time-period-based features
    for func in vintage_functions:
        # Determine which data source to use
        if 'previous_application' in func.__code__.co_varnames:
            data_source = previous_application
            storage_dict = previous_application_features
        elif 'bureau' in func.__code__.co_varnames:
            data_source = bureau
            storage_dict = bureau_features
        else:
            data_source = application_train
            storage_dict = application_features

        # Generate feature DataFrame
        feature_df = func(data_source)

        # Store the generated DataFrame in the appropriate dictionary
        feature_name = f"{func.__name__}"
        storage_dict[feature_name] = feature_df

    # Merge the feature DataFrames into their respective original dataframes
    # This allows you to have three separate dataframes: `application_train_final`, `previous_application_final`, `bureau_final`

    # Start by copying the base dataframes
    application_train_final = application_train[['SK_ID_CURR']]
    previous_application_final = previous_application[['SK_ID_CURR']]
    bureau_final = bureau[['SK_ID_CURR']]

    # Merge each DataFrame from the corresponding dictionary
    for feature_name, feature_df in application_features.items():
        application_train_final = application_train_final.merge(feature_df, on='SK_ID_CURR', how='left')

    for feature_name, feature_df in previous_application_features.items():
        previous_application_final = previous_application_final.merge(feature_df, on='SK_ID_CURR', how='left')

    for feature_name, feature_df in bureau_features.items():
        bureau_final = bureau_final.merge(feature_df, on='SK_ID_CURR', how='left')


    application_train_final = application_train_final.merge(
        application_train[['SK_ID_CURR', 'TARGET']],
        on='SK_ID_CURR',
        how='left'
    )

    #dropping duplicates of previous_application_final and bureau_final
    previous_application_final = previous_application_final.drop_duplicates()
    bureau_final = bureau_final.drop_duplicates()


    previous_application_final = previous_application_final.merge(
        application_train[['SK_ID_CURR', 'TARGET']],
        on='SK_ID_CURR',
        how='inner'
    )

    bureau_final = bureau_final.merge(
        application_train[['SK_ID_CURR', 'TARGET']],
        on='SK_ID_CURR',
        how='inner'
    )

    # First, drop duplicate TARGETs before merging
    previous_application_final = previous_application_final.drop(columns=['TARGET'])
    bureau_final = bureau_final.drop(columns=['TARGET'])

    # Merge all three on SK_ID_CURR using inner joins
    final_vintage_features_df = application_train_final.merge(previous_application_final, on='SK_ID_CURR', how='inner') \
                                       .merge(bureau_final, on='SK_ID_CURR', how='inner')

    # Final DataFrame with one SK_ID_CURR and one TARGET column
    print(final_vintage_features_df.shape)
    final_vintage_features_df.head()

    # # Save to CSV
    # final_vintage_features_df.to_csv('data/processed/training_vintage_features_df.csv', index=False)

    return final_vintage_features_df

if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parents[1]

    application_path = os.path.join(root_dir, "data", "application_train.csv")
    bureau_path = os.path.join(root_dir, "data", "bureau.csv")
    prev_application_path = os.path.join(root_dir, "data", "previous_application.csv")


    app_train = pd.read_csv(application_path)
    bureau = pd.read_csv(bureau_path)
    prev_app = pd.read_csv(prev_application_path)

    final_vintage_features_df = create_vintage_features(app_train,prev_app,bureau)

    # save in data>processed
    output_path = os.path.join(root_dir, "data", "vintage_features_1.csv")
    final_vintage_features_df.to_csv(output_path, index=False)
