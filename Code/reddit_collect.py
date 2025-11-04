"""
Need to create a .env file with your Reddit credentials:
REDDIT_CLIENT_ID=xxxx
REDDIT_CLIENT_SECRET=yyyy
REDDIT_USER_AGENT=mbta-sentiment-project (by u/your_username)
"""

# Code/reddit_collect.py
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import praw
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env if present
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "Data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

def make_reddit_client():
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "mbta-sentiment-project")

    if not client_id or not client_secret:
        raise RuntimeError("Missing Reddit credentials in environment variables.")

    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )

def utc_to_dt(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)

def fetch_posts(
    reddit,
    subreddits=("mbta", "boston"),
    start_dt=None,
    end_dt=None,
    keywords=None,
    limit=None,
):
    """
    Fetch posts from the given subreddits, filtering by time and optional keywords.
    """
    start_ts = start_dt.timestamp() if start_dt else None
    end_ts = end_dt.timestamp() if end_dt else None
    keywords = [k.lower() for k in (keywords or [])]

    rows = []
    for sub_name in subreddits:
        subreddit = reddit.subreddit(sub_name)
        print(f"🔍 Fetching posts from r/{sub_name} (new)")

        for post in tqdm(subreddit.new(limit=limit)):
            created = post.created_utc
            if start_ts and created < start_ts:
                # Since .new is reverse-chronological, we can break once older than start
                continue
            if end_ts and created > end_ts:
                continue

            # optional keyword filter in title+body
            text = (post.title or "") + " " + (post.selftext or "")
            if keywords:
                lower_text = text.lower()
                if not any(k in lower_text for k in keywords):
                    continue

            rows.append(
                {
                    "id": post.id,
                    "author": str(post.author) if post.author else "[deleted]",
                    "title": post.title,
                    "body": post.selftext,
                    "score": post.score,
                    "created_utc": created,
                    "subreddit": sub_name,
                }
            )

    return pd.DataFrame(rows)

def main():
    reddit = make_reddit_client()

    # EXAMPLE: last 12 months; adjust to match your actual project dates
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt.replace(year=end_dt.year - 1)

    keywords = [
        "mbta", "t", "green line", "red line", "orange line",
        "blue line", "commuter rail", "shuttle bus", "delay",
        "signal problem", "shutdown",
    ]

    df = fetch_posts(
        reddit,
        subreddits=("mbta", "boston"),
        start_dt=start_dt,
        end_dt=end_dt,
        keywords=keywords,
        limit=None,  # or set to something small while testing
    )

    out_path = RAW_DATA_DIR / "reddit_mbta_raw.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Saved {len(df)} posts to {out_path.relative_to(PROJECT_ROOT)}")

if __name__ == "__main__":
    main()
