import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import argparse



model2category = {
    # non instruction tuned
    "BM25": "No Instruction Training",
    "E5-base-v2": "No Instruction Training",
    "E5-large-v2": "No Instruction Training",
    "Contriever": "No Instruction Training",
    "MonoBERT": "No Instruction Training",
    "MonoT5-base": "No Instruction Training",
    "MonoT5-3B": "No Instruction Training",
    "Cohere v3 English": "Instruct-Tuned LLMs and APIs",
    "OpenAI v3 Large": "Instruct-Tuned LLMs and APIs",
    "Google Gecko": "Instruct-Tuned LLMs and APIs",
    # now these are instruction-tuned in some sense
    "E5-mistral": "Uses Instructions in Training",
    "TART-Contriever": "Uses Instructions in Training",
    "INSTRUCTOR-base": "Uses Instructions in Training",
    "INSTRUCTOR-xl": "Uses Instructions in Training",
    "E5-mistral": "Uses Instructions in Training",
    "BGE-base": "Uses Instructions in Training",
    "BGE-large": "Uses Instructions in Training",
    "TART-FLAN-T5-xl": "Uses Instructions in Training",
    "GritLM-7B": "Uses Instructions in Training",
    "FLAN-T5-base": "Instruct-Tuned LLMs and APIs",
    "FLAN-T5-large": "Instruct-Tuned LLMs and APIs",
    # LLMs/large rerankers
    "Llama-2-7B": "Instruct-Tuned LLMs and APIs",
    "Llama-2-7B-chat": "Instruct-Tuned LLMs and APIs",
    "GritLM-Reranker": "Instruct-Tuned LLMs and APIs",
    "Mistral-7B-instruct": "Instruct-Tuned LLMs and APIs",
    "FollowIR-7B": "Instruct-Tuned LLMs and APIs",

}

def make_ablation_plot(args):
    df = pd.read_csv(args.file, header=0, index_col=None)
    df = df[df.dataset.apply(lambda x: "robust" in x.lower())]
    df["category"] = df.model.apply(lambda x: model2category[x])

    # change TART models to just be TART-FLAN and TART-dual
    df["model"] = df["model"].apply(lambda x: x.replace("TART-Contriever", "TART-dual").replace("TART-FLAN-T5-xl", "TART-FLAN"))
    # make INSTRUCTOR-base to just be INSTRUCTOR-b and INSTRUCTOR-xl to INSTRUCTOR-x
    df["model"] = df["model"].apply(lambda x: x.replace("INSTRUCTOR-base", "INSTRUCTOR-b"))#.replace("INSTRUCTOR-xl", "INSTRUCTOR-x"))

    # Melt the DataFrame for grouped barplot preparation
    df_melted = df.melt(id_vars=["dataset", "model", "category"], 
                        value_vars=["keywords_score", "short_query_score", "full_score"], 
                        var_name="score_type", value_name="score")

    categories = ["No Instruction Training", "Uses Instructions in Training", "Instruct-Tuned LLMs and APIs"]
    # Set figsize width to maximize page width utilization, and reduce height per subplot
    fig, axes = plt.subplots(len(categories), 1, figsize=(12, 9), squeeze=False)
    
    for i, category in enumerate(categories):
        cur_df = df_melted[df_melted['category'] == category]
        # sort by score
        cur_df = cur_df.sort_values(by="score", ascending=False)
        ax = sns.barplot(data=df_melted[df_melted['category'] == category], x="model", y="score", 
                         hue="score_type", ax=axes[i][0], palette=sns.color_palette("ch:s=.25,rot=-.25", as_cmap=False), dodge=True)
        ax.set_title(category, fontsize=16)
        # no x-axis subtitle
        ax.set_xlabel('')
        ax.tick_params(axis='y', labelsize=13)
        # set x-axis labels to be labelsize 13
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center", fontsize=12)
        # This line can be adjusted or commented out if unnecessary
        ax.set_ylabel('Δ p-MRR compared to using the query only', fontsize=15, labelpad=15)
        if i != 1:  # Optionally, clear y-labels for non-first subplots
            ax.set_ylabel('')

    # Placing the legend at the bottom of the figure
    handles, labels = ax.get_legend_handles_labels()
    # add some spacing before the legend, and make it a little to the right
    fig.legend(handles, ["Keywords", "Short Instruction", "Full Instruction"], loc='lower center', ncol=3, fontsize=13, title="Instruction Setting", title_fontsize=15, bbox_to_anchor=(0.55, 0))

    # Removing legends from individual subplots to use the central legend
    for ax_row in axes:
        ax_row[0].get_legend().remove()
    
    # Adjust the rect to prevent overlap with the bottom legeng
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    # Save the plot with adjusted dimensions and formatting
    plt.savefig(args.file.replace(".csv", "_scientific.png"), dpi=300)
    plt.savefig(args.file.replace(".csv", "_scientific.pdf"), dpi=300)
    print(f"Saved to {args.file.replace('.csv', '_scientific.png')} and {args.file.replace('.csv', '_scientific.pdf')}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="results/ablation_results.csv", type=str)
    args = parser.parse_args()
    make_ablation_plot(args)
    # python scripts/make_ablation_plots.py --file results/ablation_results.csv