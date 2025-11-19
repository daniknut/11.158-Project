"""
sentiment_analysis.py

Wrapper script for the 11.158 MBTA Reddit project that runs:

    mbta_complaints.txt
        → topic modeling (BERTopic via get_posts_topics_bert.py)
        → transformer-based sentiment analysis (Twitter RoBERTa)
        → topic-level severity scores
        → basic visualizations saved to disk

Assumptions
-----------
- You have a sibling module `get_posts_topics_bert.py` that defines:
    - parse_mbta_complaints_txt(file_path) -> DataFrame
    - get_post_topics(df, reduce_to=None, save_model=None, return_model=False) -> DataFrame (or (DataFrame, model) if return_model=True)

- The DataFrame returned by get_post_topics includes at least:
    - 'reddit post id'
    - 'reddit title'
    - 'reddit body'
    - 'topic_id'

- Engagement fields (e.g., upvotes / comments) may or may not exist.
  If not present, severity will effectively be driven by sentiment alone.
"""

import argparse
import os
import math

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Import parsing + topic modeling from your existing script.
# Make sure get_posts_topics_bert.py is in the same directory or on PYTHONPATH.
from get_posts_topics_bert import parse_mbta_complaints_txt, get_post_topics


# ---------------------------
# Sentiment analysis helpers
# ---------------------------

SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def build_sentiment_text(
    df: pd.DataFrame, title_col: str = "reddit title", body_col: str = "reddit body"
) -> pd.Series:
    """
    Compose text for sentiment analysis by concatenating title + body.
    """
    title = df.get(title_col, "").fillna("").astype(str)
    body = df.get(body_col, "").fillna("").astype(str)
    return (title + " " + body).str.strip()


def load_sentiment_model(model_name: str = SENTIMENT_MODEL_NAME):
    """
    Load Twitter RoBERTa sentiment model and tokenizer.
    Returns (tokenizer, model, device).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[sentiment] Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def compute_sentiment_scores(
    texts,
    tokenizer,
    model,
    device: str,
    batch_size: int = 32,
    max_length: int = 256,
):
    """
    Compute sentiment probabilities and a scalar score for a list of texts.

    For cardiffnlp/twitter-roberta-base-sentiment(-latest):
        index 0 = negative
        index 1 = neutral
        index 2 = positive

    We map to a single score in [-1, 1] as:
        sentiment_score = P(positive) - P(negative)
    """
    all_neg = []
    all_neu = []
    all_pos = []
    all_score = []

    n = len(texts)
    for start in range(0, n, batch_size):
        batch_texts = texts[start : start + batch_size]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)

        neg = probs[:, 0].cpu().numpy()
        neu = probs[:, 1].cpu().numpy()
        pos = probs[:, 2].cpu().numpy()
        score = pos - neg  # [-1, 1] approximately

        all_neg.extend(neg.tolist())
        all_neu.extend(neu.tolist())
        all_pos.extend(pos.tolist())
        all_score.extend(score.tolist())

        print(f"[sentiment] Processed {min(start + batch_size, n)}/{n} posts", end="\r")

    print()  # newline after progress
    return all_neg, all_neu, all_pos, all_score


def attach_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sentiment columns to the post-level DataFrame:
        - sentiment_text
        - sent_neg_prob, sent_neu_prob, sent_pos_prob
        - sentiment_score in [-1, 1]
    """
    df = df.copy()
    df["sentiment_text"] = build_sentiment_text(df)

    tokenizer, model, device = load_sentiment_model()
    texts = df["sentiment_text"].tolist()

    print(f"[sentiment] Scoring sentiment for {len(texts)} posts...")
    neg, neu, pos, score = compute_sentiment_scores(texts, tokenizer, model, device)

    df["sent_neg_prob"] = neg
    df["sent_neu_prob"] = neu
    df["sent_pos_prob"] = pos
    df["sentiment_score"] = score

    return df


# ---------------------------
# Severity + aggregation
# ---------------------------


# def infer_engagement(df: pd.DataFrame) -> pd.Series:
#     """
#     Heuristic engagement metric per post.

#     Tries to use any available columns that look like:
#         - 'score' or 'reddit score'
#         - 'num_comments' or 'reddit num comments'
#         - 'upvotes', 'downs', etc.

#     If none are present, returns a constant 1 for all posts so
#     that severity collapses to (rescaled) mean sentiment.
#     """
#     # Start with zeros
#     engagement = pd.Series(0.0, index=df.index)

#     candidate_cols = [
#         "score",
#         "reddit score",
#         "ups",
#         "downs",
#         "num_comments",
#         "num comments",
#         "comment_count",
#         "upvote_ratio",
#         "engagement",  # if user already defined it
#     ]

#     used_any = False
#     for col in candidate_cols:
#         if col in df.columns:
#             vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
#             # Ensure non-negative contribution
#             vals = vals.abs()
#             engagement = engagement + vals
#             used_any = True

#     if not used_any:
#         # Fallback: constant engagement
#         engagement = pd.Series(1.0, index=df.index)
#         print(
#             "[severity] No explicit engagement columns found; "
#             "using engagement = 1 for all posts."
#         )
#     else:
#         print("[severity] Built engagement metric from available columns.")

#     return engagement


def compute_topic_severity(
    df: pd.DataFrame,
    topic_col: str = "topic_id",
    sentiment_col: str = "sentiment_score",
    id_col: str = "reddit post id",
) -> pd.DataFrame:
    """
    Aggregate sentiment and engagement at the topic level and compute a
    severity score:

        Topic Severity = mean_sentiment * (1 + log(1 + mean_engagement))

    Returns a DataFrame with:
        topic_id, num_posts, mean_sentiment, median_sentiment, sentiment_std,
        mean_engagement, topic_severity
    """
    if topic_col not in df.columns:
        raise ValueError(f"{topic_col} not in DataFrame; run topic modeling first.")
    if sentiment_col not in df.columns:
        raise ValueError(
            f"{sentiment_col} not in DataFrame; run sentiment scoring first."
        )

    # Build per-post engagement
    df = df.copy()
    # df["engagement"] = infer_engagement(df)
    engagement = pd.Series(1.0, index=df.index)
    df["engagement"] = engagement

    grouped = (
        df.groupby(topic_col)
        .agg(
            num_posts=(id_col, "count"),
            mean_sentiment=(sentiment_col, "mean"),
            median_sentiment=(sentiment_col, "median"),
            sentiment_std=(sentiment_col, "std"),
            mean_engagement=("engagement", "mean"),
        )
        .reset_index()
    )

    # Attach human-readable labels if they exist in the original DataFrame
    label_cols = []
    for col in ["topic_label", "topic_keywords"]:
        if col in df.columns:
            label_cols.append(col)
    if label_cols:
        label_df = df[[topic_col] + label_cols].drop_duplicates(subset=[topic_col])
        grouped = grouped.merge(label_df, on=topic_col, how="left")

    # Compute severity metric
    grouped["topic_severity"] = grouped["mean_sentiment"] * (
        1.0 + np.log1p(grouped["mean_engagement"])
    )

    return grouped


# ---------------------------
# Visualization helpers
# ---------------------------


def plot_sentiment_bar(
    topic_df: pd.DataFrame,
    out_path: str,
    sort_by: str = "topic_severity",
    top_n: int = 10,
):
    """
    Bar chart of mean sentiment per topic, labeled with topic_label when available,
    showing only the top_n topics by the chosen sort metric.
    """
    # Take the top N by severity (or whatever sort_by is)
    topic_df = topic_df.sort_values(sort_by, ascending=False).head(top_n)

    # Build human-readable x labels:
    # Prefer topic_label, then fall back to "id: keywords", then just topic_id.
    if "topic_label" in topic_df.columns:
        x_labels = topic_df["topic_label"].astype(str)
    elif "topic_keywords" in topic_df.columns:
        x_labels = topic_df.apply(
            lambda r: f"{int(r['topic_id'])}: {r['topic_keywords']}",
            axis=1,
        )
    else:
        x_labels = topic_df["topic_id"].astype(str)

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(topic_df)), topic_df["mean_sentiment"])
    plt.xticks(range(len(topic_df)), x_labels, rotation=60, ha="right")
    plt.xlabel("Topic")
    plt.ylabel("Mean sentiment score (-1 to 1)")
    plt.title(f"Top {top_n} Topics by {sort_by.replace('_', ' ').title()}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[viz] Saved bar chart to {out_path}")


# def plot_severity_scatter(
#     topic_df: pd.DataFrame,
#     out_path: str,
# ):
#     """
#     Scatter plot of mean sentiment vs mean engagement, bubble size ~ num_posts,
#     color ~ topic_severity sign.
#     """
#     plt.figure(figsize=(8, 6))

#     x = topic_df["mean_sentiment"]
#     y = topic_df["mean_engagement"]
#     max_posts = max(topic_df["num_posts"].max(), 1)
#     sizes = 20 + 80 * (topic_df["num_posts"] / max_posts)

#     # Color points by severity (red = more severe negative, blue = positive)
#     cmap = plt.get_cmap("coolwarm")
#     # Normalize severity to [-1, 1] range for coloring (clamp extremes)
#     sev = topic_df["topic_severity"]
#     sev_norm = np.clip(sev / (np.abs(sev).max() if np.abs(sev).max() > 0 else 1), -1, 1)
#     colors = cmap((sev_norm + 1) / 2.0)

#     plt.scatter(x, y, s=sizes, c=colors, alpha=0.8, edgecolors="k")
#     plt.axvline(0, color="gray", linestyle="--", linewidth=1)
#     plt.xlabel("Mean sentiment score (-1 to 1)")
#     plt.ylabel("Mean engagement")
#     plt.title("Topic-level Sentiment vs Engagement")
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=300)
#     plt.close()
#     print(f"[viz] Saved scatter plot to {out_path}")


# ---------------------------
# End-to-end wrapper
# ---------------------------


def run_pipeline(
    txt_path: str,
    reduce_to: int | None,
    data_dir: str = "../Data",
    fig_dir: str = "../Figures",
):
    """
    Full wrapper: topics → sentiment → severity → visualizations.
    """
    print(f"[pipeline] Parsing complaints from {txt_path} ...")
    df_raw = parse_mbta_complaints_txt(txt_path)
    print(f"[pipeline] Parsed {len(df_raw)} posts.")

    # Topic modeling
    print("[pipeline] Running BERTopic (via get_post_topics_bert)...")
    # We do not reduce topics on the first pass so we keep detail; if reduce_to is
    # provided, we run a second pass to get a compact topic structure if needed.
    if reduce_to is not None:
        df_topics, topic_model = get_post_topics(
            df_raw, reduce_to=reduce_to, return_model=True
        )
    else:
        df_topics, topic_model = get_post_topics(
            df_raw, reduce_to=None, return_model=True
        )

    # Save intermediate topic output (with body kept)
    posts_topics_path = os.path.join(data_dir, "mbta_posts_with_topics_full.csv")
    df_topics.to_csv(posts_topics_path, index=False)
    print(f"[pipeline] Saved posts with topics → {posts_topics_path}")

    # Sentiment scoring
    print("[pipeline] Attaching sentiment scores...")
    df_with_sentiment = attach_sentiment(df_topics)

    posts_sent_path = os.path.join(data_dir, "mbta_posts_with_topics_and_sentiment.csv")
    df_with_sentiment.to_csv(posts_sent_path, index=False)
    print(f"[pipeline] Saved posts with topics + sentiment → {posts_sent_path}")

    # Topic-level aggregation + severity
    print("[pipeline] Aggregating topic-level severity...")
    topic_stats = compute_topic_severity(df_with_sentiment)

    # Sort by most severe topics first to make the CSV easier to read
    topic_stats = topic_stats.sort_values("topic_severity", ascending=False)

    topic_stats_path = os.path.join(data_dir, "mbta_topic_sentiment_severity.csv")
    topic_stats.to_csv(topic_stats_path, index=False)
    print(f"[pipeline] Saved topic-level sentiment + severity → {topic_stats_path}")

    # Visualizations
    bar_path = os.path.join(fig_dir, "topic_mean_sentiment_bar.png")
    plot_sentiment_bar(topic_stats, bar_path, sort_by="topic_severity")

    # scatter_path = os.path.join(fig_dir, "topic_sentiment_vs_engagement_scatter.png")
    # plot_severity_scatter(topic_stats, scatter_path)

    print("[pipeline] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MBTA Reddit topics → sentiment → severity → visualizations pipeline."
    )
    parser.add_argument(
        "--txt_path",
        type=str,
        default="mbta_complaints.txt",
        help="Path to mbta_complaints.txt",
    )
    parser.add_argument(
        "--reduce_to",
        type=int,
        default=None,
        help="Optional number of topics to reduce to in BERTopic.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../Data",
        help="Directory to save CSV outputs.",
    )
    parser.add_argument(
        "--fig_dir",
        type=str,
        default="../Figures",
        help="Directory to save figure outputs.",
    )
    args = parser.parse_args()

    run_pipeline(
        txt_path=args.txt_path,
        reduce_to=args.reduce_to,
        data_dir=args.data_dir,
        fig_dir=args.fig_dir,
    )
