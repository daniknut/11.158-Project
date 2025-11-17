import argparse
import os
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, PartOfSpeech, MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

def load_docs(df, text_column, title_column=None, body_column=None):
    # Compose document text from available columns (title + body) or a single text column
    
    if text_column and text_column in df.columns:
        return df[text_column].fillna("").astype(str).tolist()
    parts = []
    for _, row in df.iterrows():
        txt = ""
        if title_column and title_column in df.columns:
            txt += str(row.get(title_column, "")) + " "
        if body_column and body_column in df.columns:
            txt += str(row.get(body_column, ""))
        parts.append(txt.strip())
    return parts


def build_topic_model():
    # Main representation and additional aspects	
    """
    Build a BERTopic model with a main representation based on KeyBERTInspired.
        
    The model will include two additional aspects:
    - Aspect 1: PartOfSpeech (using the spaCy 'en_core_web_sm' model) or
    fallback to KeyBERTInspired if spaCy is not available

    - Aspect 2: a combination ofKeyBERTInspired (with top_n_words=30) and
    MaximalMarginalRelevance (with diversity=.5).

    The model will use a lightweight sentence transformer for embeddings.

    Returns a BERTopic object ready to fit a set of documents and predict topics.
    """
    # Increase diversity in main representation to focus on more specific keywords
    main_representation = KeyBERTInspired()
    try:
        aspect_model1 = PartOfSpeech("en_core_web_sm")
    except Exception:
        # If spaCy model isn't available, fallback to KeyBERTInspired for stability
        aspect_model1 = KeyBERTInspired()
    aspect_model2 = [KeyBERTInspired(top_n_words=30), MaximalMarginalRelevance(diversity=.5)]

    rep_model = {
        "Main": main_representation,
        "Aspect1": aspect_model1,
        "Aspect2": aspect_model2
    }

    # Use a lightweight sentence transformer for embeddings
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    # Add stop words to remove common MBTA-related terms and focus on specific complaints
    general_transit_words = ["mbta", "rail", "bus", "transit", "trains", "boston", "station", "train", "commuter", "going", "ride", "t", "red", "blue", "green", "orange", "silver", "line", "subway", "metro"]
    general_words = ["do", "had"]
    stop_words = general_transit_words + general_words
    vectorizer_model = CountVectorizer(stop_words=stop_words)
    topic_model = BERTopic(
        representation_model=rep_model,
        embedding_model=emb_model,
        min_topic_size=5,
        vectorizer_model=vectorizer_model
    )
    # topic_model = BERTopic(representation_model=rep_model, embedding_model=emb_model)
    return topic_model


def attach_topic_info(df, topics, topic_model):
    # Add topic id and human-readable label + top keywords	
    """
        Attach topic information to a DataFrame with BERTopic results.

        Args:
            df (pandas.DataFrame): DataFrame with BERTopic results
            topics (list): List of topic IDs
            topic_model (BERTopic): Trained BERTopic model

        Returns:
            pandas.DataFrame: DataFrame with additional columns for topic id and
            human-readable label + top keywords
    """
 
    df = df.copy()
    df["topic_id"] = topics
 
    # Get topic labels (topic_model.get_topic_info gives a summary)
    info = topic_model.get_topic_info()
 
    # Build a map from topic id to keywords string
    topic_keywords = {}
    for tid in info.Topic.unique():
        if tid == -1:
            topic_keywords[tid] = "Outlier"
            continue
        kw = topic_model.get_topic(tid)
        if not kw:
            topic_keywords[tid] = ""
        else:
            topic_keywords[tid] = ", ".join([w for w, _ in kw[:10]])
   
    df["topic_keywords"] = df["topic_id"].map(lambda t: topic_keywords.get(t, ""))
 
    # Add a label column combining id and keywords
    df["topic_label"] = df.apply(lambda r: f"{r['topic_id']}: {r['topic_keywords']}", axis=1)
    return df

def get_post_topics(df, id_col="reddit post id", title_col="reddit title", body_col="reddit body", reduce_to=None, save_model=None, return_model=False):
    """
    Annotate a DataFrame of Reddit posts with BERTopic topic assignments.

    Args:
        df (pd.DataFrame): DataFrame containing reddit posts.
        id_col (str): column name for reddit post id (must exist).
        title_col (str): column name for reddit title (optional).
        body_col (str): column name for reddit body (optional).
        reduce_to (int): optional target number of topics after fitting.
        save_model (str): optional path to save the BERTopic model (.zip).
        return_model (bool): if True, return (out_df, topic_model) else just out_df.

    Returns:
        pd.DataFrame or (pd.DataFrame, BERTopic): DataFrame with topic columns (and model if requested).
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    if id_col not in df.columns:
        raise ValueError(f"Missing id column: {id_col}")

    # Compose documents from title + body columns (force composition by passing text_column=None)
    docs = load_docs(df, text_column=None, title_column=title_col, body_column=body_col)

    topic_model = build_topic_model()
    topics, probs = topic_model.fit_transform(docs)
    print(f"Initial number of topics: {len(topic_model.get_topic_info())}")

    if reduce_to:
        print(f"Reducing topics to {reduce_to}...")
        try:
            topic_model.reduce_topics(docs, nr_topics=reduce_to)
            topics, probs = topic_model.transform(docs)
        except Exception as e:
            print(f"Could not reduce topics: {e}")

    out_df = attach_topic_info(df, topics, topic_model)

    if save_model:
        topic_model.save(save_model)
        print(f"Saved BERTopic model to {save_model}")

    if return_model:
        return out_df, topic_model
    return out_df

def parse_mbta_complaints_txt(file_path):
    """
    Parse the mbta_complaints.txt file into a pandas DataFrame.

    Args:
        file_path (str): Path to the txt file.

    Returns:
        pd.DataFrame: DataFrame with columns 'reddit post id', 'reddit title', 'reddit body'.
    """
    posts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by post separators
    post_blocks = content.split('================================================================================\nPost #')
    
    for block in post_blocks[1:]:  # Skip the header
        lines = block.split('\n')
        post_id = None
        title = None
        body = []
        in_content = False
        
        for line in lines:
            if line.startswith('Author: '):
                author = line.split('Author: ')[1].strip()
            elif line.startswith('Post ID: '):
                post_id = line.split('Post ID: ')[1].strip()
            elif line.startswith('Title: '):
                title = line.split('Title: ')[1].strip()
            elif line.startswith('Content:'):
                in_content = True
            elif in_content and line.startswith('--------------------------------------------------------------------------------'):
                continue
            elif in_content and line.strip() == '================================================================================':
                break
            elif in_content:
                body.append(line)
        
        # Exclude posts by AutoModerator and ensure required fields are present
        # TO_DISCUSS: Should we exclude AutoModerator? I assume so
        if "AutoModerator" not in author and post_id and title:
            body_text = '\n'.join(body).strip()
            posts.append({
                'reddit post id': post_id,
                'reddit title': title,
                'reddit body': body_text
            })
    
    return pd.DataFrame(posts)

def extract_topics(df, reduce_to=None):
    # Return model to get topic summary
    result_df, topic_model = get_post_topics(df, reduce_to=reduce_to, return_model=True)
    
	# TO_DISCUSS: Should we remove the body column to make it easier to read?
    # Remove the body column to make csv easier to read
    result_df = result_df.drop(columns=['reddit body'])

    # Get topic summary
    topic_summary = topic_model.get_topic_info()[['Topic', 'Count', 'Name']].rename(columns={
        'Topic': 'topic_id',
        'Count': 'num_posts',
        'Name': 'topic_keywords'
    })

    # Save the result to CSV
    topic_summary_name = "mbta_posts_with_topics reduced to " + str(reduce_to) if reduce_to else "mbta_posts_with_topics"
    output_path = r"..\Data\\" + topic_summary_name + ".csv"
    result_df.to_csv(output_path, index=False)
    print(f"Processed {len(result_df)} posts and saved to {output_path}")

    # Save topic summary to CSV
    summary_name = "mbta_topic_summary reduced to " + str(reduce_to) if reduce_to else "mbta_topic_summary"
    summary_path = r"..\Data\\" + summary_name + ".csv"
    topic_summary.to_csv(summary_path, index=False)
    print(f"Topic summary saved to {summary_path}")
    
    
if __name__ == "__main__":
    # Path to the mbta_complaints.txt file
    txt_file_path = r"..\Data\mbta_complaints.txt"
    
    # Parse the txt file into DataFrame
    df = parse_mbta_complaints_txt(txt_file_path)
    
    print(f"Parsed {len(df)} posts from {txt_file_path}")
    print(df.head())
    # Apply topic modeling
    # result_df = get_post_topics(df, reduce_to=10, save_model="mbta_topics_model.zip")
    
    # TO_DISCUSS: How many topics to reduce to?
    # Apply topic modeling with more topics for specificity
    # result_df = get_post_topics(df, reduce_to=10)  # Increased to 20 for more specific categories
    
    extract_topics(df)
    extract_topics(df, reduce_to=10)