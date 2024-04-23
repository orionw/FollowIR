import pandas as pd
import json
import os
import glob
import argparse
import tqdm
from collections import defaultdict

from gather_results import model_order, MAP_MODEL_ORDER


GRADIENT_STR = """\midpointgradientcell{VALUE}{MIN}{MAX}{0}{neg}{pos}{\opacity}{0}"""


def gather_results(args, dataset_in_table=["Robust04InstructionRetrieval", "News21InstructionRetrieval", "Core17InstructionRetrieval"]):
    # go through all in `results` and aggregate them together
    # we care only about the pairwise and rankwise scores, as well as map@1000 and ndcg@5 scores of the original and changed

    all_data = []
    for file in tqdm.tqdm(glob.glob(os.path.join(args.results_dir, "*", "*.json"))):
        dataset_name = file.split("/")[-1].replace(".json", "")
        model_name = file.split("/")[-2]
        with open(file, "r") as f:
            data = json.load(f)["test"] # all on test set

            # map@1000 and ndcg@5
            map1000 = data["individual"]["base"]["map_at_1000"]
            ndcg5 = data["individual"]["base"]["ndcg_at_5"]

            # map@1000 and ndcg@5
            map1000_full = data["individual"]["original"]["map_at_1000"]
            ndcg5_full = data["individual"]["original"]["ndcg_at_5"]

            # map@1000 and ndcg@5 of the short_query
            map1000_short = data["length_ablation"]["short_instructions"][0]["map_at_1000"]
            ndcg5_short = data["length_ablation"]["short_instructions"][0]["ndcg_at_5"]

            # map@100 and ndcg@5 of the keywords
            map1000_keywords = data["length_ablation"]["keywords"][0]["map_at_1000"]
            ndcg5_keywords = data["length_ablation"]["keywords"][0]["ndcg_at_5"]

            # diffs
            short_diff_map = map1000_short - map1000
            short_diff_ndcg = ndcg5_short - ndcg5

            keywords_diff_map = map1000_keywords - map1000
            keywords_diff_ndcg = ndcg5_keywords - ndcg5

            full_diff_map = map1000_full - map1000
            full_diff_ndcg = ndcg5_full - ndcg5


            # add to the list
            all_data.append({
                "dataset": dataset_name,
                "model": model_name,
                "short_query_score": short_diff_map if "news" not in dataset_name.lower() else short_diff_ndcg,
                "keywords_score": keywords_diff_map if "news" not in dataset_name.lower() else keywords_diff_ndcg,
                "full_score": full_diff_map if "news" not in dataset_name.lower() else full_diff_ndcg,
                "main_score": map1000 if "news" not in dataset_name.lower() else ndcg5
            })

    keep_columns = ["keywords_score", "short_query_score", "full_score"]

    # create a dataframe
    df = pd.DataFrame(all_data)
    # sort by map and then rankwise
    df = df.sort_values(by=keep_columns, ascending=[False, False, False])

    # lets turn this into a latex figure
    # aggregate by dataset, and grab only the map (for Robust and Core) or (nDCG) of the original, plus the pointwise and rankwise scores
    df = df[df["dataset"].isin(dataset_in_table)]
    df = df.groupby(["dataset", "model"]).agg({"full_score": "first", "keywords_score": "first", "short_query_score": "first"}).reset_index()
    # for every metric, multiply by 100 and round and format to nearest tenth
    for col in keep_columns:
        df[col] = (df[col] * 100).round(1).astype(str)

    # order by model
    df["model"] = df["model"].map(MAP_MODEL_ORDER)
    # keep only the ones that are not nan, e.g. not in the model
    df = df[df.model.notna()]
    df["model"] = pd.Categorical(df["model"], model_order.keys())
    df = df.sort_values(by=["model"])
    df.to_csv(os.path.join(args.results_dir, "ablation_results.csv"), index=False)
    # First, pivot your DataFrame to get 'model' as index and have a multi-level column with 'dataset' and the scores
    pivoted_df = df.pivot(index='model', columns='dataset')
    # Flatten the MultiIndex in columns, concatenating level values
    pivoted_df.columns = [' '.join(col).strip() for col in pivoted_df.columns.values]
    # Since you seem to wish for a specific order and naming of columns, you can explicitly reindex/organize
    # This step might need to be adjusted based on exact desired format, especially if you have dynamic datasets
    # The list of new column names should be formed based on the datasets and scores present in your original dataframe
    new_column_order = [
        'keywords_score Robust04InstructionRetrieval', 'short_query_score Robust04InstructionRetrieval','full_score Robust04InstructionRetrieval', 
        'keywords_score News21InstructionRetrieval', 'short_query_score News21InstructionRetrieval','full_score News21InstructionRetrieval',
        'keywords_score Core17InstructionRetrieval', 'short_query_score Core17InstructionRetrieval', 'full_score Core17InstructionRetrieval',
    ]
    pivoted_and_ordered_df = pivoted_df.reindex(columns=new_column_order).reset_index()
    # Now, your DataFrame is in the desired format. If you want to rename the columns for appearance,
    # you can manually set them as follows (This step is optional and should be customized based on your needs)

    # for each value, if "-" not in it, add "+"
    x_format = lambda x: str(x) if "-" in str(x) else "+"+str(x)
    for i, col in enumerate(pivoted_and_ordered_df.columns):
        if "model" not in col:

            pivoted_and_ordered_df[col] = pivoted_and_ordered_df[col].apply(lambda x: x_format(x))


    # Print or return your rearranged DataFrame
    print(pivoted_and_ordered_df)
    # add a column at the beginning that is empty
    pivoted_and_ordered_df.insert(0, "empty", "")
    
    pivoted_and_ordered_df.to_latex(os.path.join(args.results_dir, "ablation_results.tex"), index=False)
    print(f"Saved to {os.path.join(args.results_dir, 'ablation_results.tex')}")
    print(f"Saved to {os.path.join(args.results_dir, 'ablation_results.csv')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results", type=str)
    args = parser.parse_args()
    gather_results(args)
            