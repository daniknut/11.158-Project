# Data Dictionary

This file documents all datasets, variables, units, and basic cleaning/filtering steps.

## reddit_mbta_raw.csv
- `id`: Reddit post ID
- `author`: Anonymized user name or ID
- `title`: Post title
- `body`: Post selftext
- `score`: Upvotes (net)
- `created_utc`: Unix timestamp (seconds) in UTC
- `subreddit`: Subreddit name

## mbta_alerts_raw.csv
- `alert_id`
- `route`
- `effect`
- `severity`
- `description`
- `start_time`
- `end_time`

## weather_boston_raw.csv
- `timestamp`
- `temp_c`
- `precip_mm`
- `condition` (e.g., rain / snow / clear)

(Add to and refine this as you go!)
