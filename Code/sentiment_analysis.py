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
    print(topic_df)
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


def plot_sentiment_histogram(
    df: pd.DataFrame,
    out_path: str,
    sentiment_col: str = "sentiment_score",
    bins: int = 30,
) -> None:
    """
        Plot a histogram of post-level sentiment scores.

        This shows how positive/negative the overall corpus is and can be used
        as a quick sanity check in the report (e.g., most posts are negative).
        """
    if sentiment_col not in df.columns:
        raise ValueError(
            f"{sentiment_col} not found in DataFrame; run sentiment scoring first."
        )

    scores = df[sentiment_col].dropna().values
    if scores.size == 0:
        raise ValueError("No sentiment scores available to plot.")

    plt.figure(figsize=(8, 5))
    plt.hist(scores, bins=bins, edgecolor="black", alpha=0.8)
    plt.xlabel("Sentiment score (-1 to 1)")
    plt.ylabel("Number of posts")
    plt.title("Distribution of post-level sentiment scores")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[viz] Saved sentiment histogram to {out_path}")


def plot_severity_scatter(
    topic_df: pd.DataFrame,
    out_path: str,
    x_col: str = "mean_sentiment",
    y_col: str = "mean_engagement",
    size_col: str = "num_posts",
    color_by: str = "topic_severity",
) -> None:
    """
        Scatter plot of topic-level sentiment vs engagement (or any two metrics).

        Points are sized by the number of posts per topic and colored by
        topic severity by default. This is a nice 2D summary to drop into the
        report when describing which issues are both common and emotionally charged.
        """
    for col in [x_col, y_col, size_col, color_by]:
        if col not in topic_df.columns:
            raise ValueError(f"{col} not found in topic_df; check aggregation step.")

    df_plot = topic_df.copy()

    # Build human-readable labels similar to the bar chart
    if "topic_label" in df_plot.columns:
        labels = df_plot["topic_label"].astype(str)
    elif "topic_keywords" in df_plot.columns:
        labels = df_plot.apply(
            lambda r: f"{int(r['topic_id'])}: {r['topic_keywords']}", axis=1
        )
    else:
        labels = df_plot["topic_id"].astype(str)

    x = df_plot[x_col].values
    y = df_plot[y_col].values
    sizes_raw = df_plot[size_col].values.astype(float)
    # Normalize marker sizes for readability
    if sizes_raw.max() > 0:
        sizes = 50 + 250 * (sizes_raw / sizes_raw.max())
    else:
        sizes = np.full_like(sizes_raw, 100.0)

    c = df_plot[color_by].values

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x, y, s=sizes, c=c, cmap="viridis", alpha=0.8)
    plt.xlabel(x_col.replace("_", " ").title())
    plt.ylabel(y_col.replace("_", " ").title())
    plt.title("Topic-level sentiment vs engagement")

    cbar = plt.colorbar(scatter)
    cbar.set_label(color_by.replace("_", " ").title())

    # Light annotation for the most prominent topics (by severity)
    try:
        top_idx = np.argsort(-df_plot[color_by].values)[:10]
        for idx in top_idx:
            plt.annotate(
                labels.iloc[idx],
                (x[idx], y[idx]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )
    except Exception:
        # If something goes wrong with annotation, we still want the core plot.
        pass

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[viz] Saved severity scatter to {out_path}")


def plot_topic_mean_sentiment_histogram(
    topic_df: pd.DataFrame,
    out_path: str,
    sentiment_col: str = "mean_sentiment",
    bins: int = 20,
) -> None:
    """
    Histogram of topic-level mean sentiment values.

    This is useful for describing how many topics are near zero vs strongly
    negative, as in the narrative for Section 4.3.
    """
    if sentiment_col not in topic_df.columns:
        raise ValueError(f"{sentiment_col} not found in topic_df; check aggregation step.")

    vals = topic_df[sentiment_col].dropna().values
    if vals.size == 0:
        raise ValueError("No topic-level sentiment values available to plot.")

    plt.figure(figsize=(8, 5))
    plt.hist(vals, bins=bins, edgecolor="black", alpha=0.8)
    plt.xlabel("Mean topic sentiment (-1 to 1)")
    plt.ylabel("Number of topics")
    plt.title("Distribution of topic-level mean sentiment")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[viz] Saved topic mean sentiment histogram to {out_path}")


def plot_severity_bar(
    topic_df: pd.DataFrame,
    out_path: str,
    top_n: int = 20,
) -> None:
    """
    Bar chart of topic severity for the top_n most severe topics.

    Here, "most severe" is interpreted as the most negative severity scores.
    This directly supports the Section 4.4 severity ranking discussion.
    """
    if "topic_severity" not in topic_df.columns:
        raise ValueError("topic_severity not found in topic_df; run severity aggregation first.")

    # Most severe topics have the most negative severity scores.
    df_sorted = topic_df.sort_values("topic_severity", ascending=True).head(top_n)

    # Build labels consistent with other plots
    if "topic_label" in df_sorted.columns:
        x_labels = df_sorted["topic_label"].astype(str)
    elif "topic_keywords" in df_sorted.columns:
        x_labels = df_sorted.apply(
            lambda r: f"{int(r['topic_id'])}: {r['topic_keywords']}", axis=1,
        )
    else:
        x_labels = df_sorted["topic_id"].astype(str)

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(df_sorted)), df_sorted["topic_severity"])
    plt.xticks(range(len(df_sorted)), x_labels, rotation=60, ha="right")
    plt.xlabel("Topic")
    plt.ylabel("Topic severity")
    plt.title(f"Top {top_n} most severe topics (by severity score)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[viz] Saved severity bar chart to {out_path}")


def compute_sentiment_band_counts(
    topic_df: pd.DataFrame,
    sentiment_col: str = "mean_sentiment",
) -> dict:
    """
    Compute counts of topics in three sentiment bands used in the writeup:

    - near_zero_or_positive: mean_sentiment > -0.05
    - mildly_negative: -0.40 <= mean_sentiment <= -0.05
    - strongly_negative: mean_sentiment < -0.40

    Returns a dict with counts and proportions you can paste into the report.
    """
    if sentiment_col not in topic_df.columns:
        raise ValueError(f"{sentiment_col} not found in topic_df; check aggregation step.")

    vals = topic_df[sentiment_col]
    n_total = len(vals)

    near_zero_or_positive = (vals > -0.05).sum()
    mildly_negative = ((vals <= -0.05) & (vals >= -0.40)).sum()
    strongly_negative = (vals < -0.40).sum()

    summary = {
        "n_topics": int(n_total),
        "near_zero_or_positive": int(near_zero_or_positive),
        "mildly_negative": int(mildly_negative),
        "strongly_negative": int(strongly_negative),
        "near_zero_or_positive_prop": float(near_zero_or_positive / n_total) if n_total else 0.0,
        "mildly_negative_prop": float(mildly_negative / n_total) if n_total else 0.0,
        "strongly_negative_prop": float(strongly_negative / n_total) if n_total else 0.0,
    }

    print("[summary] Sentiment band counts:", summary)
    return summary


def extract_example_posts_for_topics(
    df_posts: pd.DataFrame,
    topic_ids,
    n_posts_per_topic: int = 3,
    topic_col: str = "topic_id",
    sentiment_col: str = "sentiment_score",
) -> pd.DataFrame:
    """
    For a given list of topic IDs, return up to n_posts_per_topic of the most
    negative posts (lowest sentiment_score) per topic.

    This is useful for pulling concrete example posts for the narrative in 4.3
    (e.g., odors/fumes, Fitchburg line delays, accessibility issues).
    """
    missing_cols = [c for c in [topic_col, sentiment_col] if c not in df_posts.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in df_posts: {missing_cols}")

    frames = []
    for tid in topic_ids:
        sub = df_posts[df_posts[topic_col] == tid].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(sentiment_col, ascending=True).head(n_posts_per_topic)
        sub["example_topic_id"] = tid
        frames.append(sub)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, axis=0)

    # Keep some useful columns if they exist
    keep_cols = [
        c
        for c in [
            topic_col,
            "example_topic_id",
            "reddit post id",
            "reddit title",
            "reddit body",
            "topic_label",
            "topic_keywords",
            sentiment_col,
        ]
        if c in result.columns
    ]
    return result[keep_cols]


def extract_top_negative_topics_and_posts(
    df_posts: pd.DataFrame,
    topic_df: pd.DataFrame,
    n_topics: int = 5,
    n_posts_per_topic: int = 3,
    topic_col: str = "topic_id",
    topic_severity_col: str = "topic_severity",
    sentiment_col: str = "sentiment_score",
):
    """
    Convenience wrapper: pick the n_topics most severe (most negative) topics
    and then extract the n_posts_per_topic most negative posts within each.

    Returns (top_topics_df, example_posts_df).
    """
    if topic_severity_col not in topic_df.columns:
        raise ValueError("topic_severity not found in topic_df; run severity aggregation first.")

    # Most severe topics have the most negative severity scores.
    top_topics = topic_df.sort_values(topic_severity_col, ascending=True).head(n_topics)
    topic_ids = top_topics[topic_col].tolist()

    example_posts = extract_example_posts_for_topics(
        df_posts,
        topic_ids=topic_ids,
        n_posts_per_topic=n_posts_per_topic,
        topic_col=topic_col,
        sentiment_col=sentiment_col,
    )

    return top_topics, example_posts

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

    # Additional visualizations for the report
    hist_path = os.path.join(fig_dir, "post_sentiment_histogram.png")
    plot_sentiment_histogram(df_with_sentiment, hist_path)

        # Visualizations
    # 1) Bar chart of topics by severity (for severity discussion)
    bar_path = os.path.join(fig_dir, "topic_mean_sentiment_bar.png")
    plot_sentiment_bar(topic_stats, bar_path, sort_by="topic_severity")

    # 1b) Bar chart of topics sorted by mean sentiment (for Section 4.3)
    mean_sent_bar_path = os.path.join(fig_dir, "topic_mean_sentiment_bar_by_mean.png")
    plot_sentiment_bar(
        topic_stats,
        mean_sent_bar_path,
        sort_by="mean_sentiment",
        top_n=len(topic_stats),
    )

    # 2) Post-level sentiment histogram (already in your draft)
    hist_path = os.path.join(fig_dir, "post_sentiment_histogram.png")
    plot_sentiment_histogram(df_with_sentiment, hist_path)

    # 2b) Topic-level mean sentiment histogram (for “Sentiment varies widely…”)
    topic_hist_path = os.path.join(fig_dir, "topic_mean_sentiment_histogram.png")
    plot_topic_mean_sentiment_histogram(topic_stats, topic_hist_path)

    # 4) Severity bar chart for top-20 most severe topics (Section 4.4)
    severity_bar_path = os.path.join(fig_dir, "topic_severity_bar_top20.png")
    plot_severity_bar(topic_stats, severity_bar_path, top_n=20)

    # 5) Sentiment band summary for reporting (near-zero, mildly negative, strongly negative)
    band_summary = compute_sentiment_band_counts(topic_stats)
    band_summary_path = os.path.join(data_dir, "mbta_topic_sentiment_band_summary.json")
    pd.Series(band_summary).to_json(band_summary_path, indent=2)
    print(f"[pipeline] Saved sentiment band summary -> {band_summary_path}")

    # 6) Save a table of top-20 most severe topics for copy/paste into the report
    top20_severe = topic_stats.sort_values("topic_severity", ascending=True).head(20)
    top_cols = [
        c
        for c in [
            "topic_id",
            "topic_label",
            "topic_keywords",
            "num_posts",
            "mean_sentiment",
            "median_sentiment",
            "sentiment_std",
            "mean_engagement",
            "topic_severity",
        ]
        if c in top20_severe.columns
    ]
    top20_path = os.path.join(data_dir, "mbta_topic_severity_top20.csv")
    top20_severe[top_cols].to_csv(top20_path, index=False)
    print(f"[pipeline] Saved top-20 most severe topics -> {top20_path}")

    # 7) Extract concrete example posts for the most severe topics (for “Example high-negative posts”)
    top_topics_df, example_posts_df = extract_top_negative_topics_and_posts(
        df_with_sentiment,
        topic_stats,
        n_topics=5,
        n_posts_per_topic=5,
    )

    examples_topics_path = os.path.join(data_dir, "mbta_example_severe_topics.csv")
    top_topics_df.to_csv(examples_topics_path, index=False)
    print(f"[pipeline] Saved metadata for example severe topics -> {examples_topics_path}")

    examples_posts_path = os.path.join(data_dir, "mbta_example_severe_posts.csv")
    example_posts_df.to_csv(examples_posts_path, index=False)
    print(f"[pipeline] Saved example posts for severe topics -> {examples_posts_path}")

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
