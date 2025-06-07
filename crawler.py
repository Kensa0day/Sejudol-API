import asyncio
from twikit import Client, TooManyRequests
from datetime import datetime
from dateutil import parser
import csv
from configparser import ConfigParser
from random import randint


# MINIMUM_TWEETS = 500
# Hanya ambil tweet berbahasa Indonesia, bukan retweet, tanpa link/media
# high_prec_phrases = [
#     "judi online", "main slot", "jackpot situs", "agen slot", "togel hari ini",
#     "judol", "casino", "togel", "taruhan", "poker","bola tangkas", "deposit judi slot", "cara daftar slot", "daftar slot", "slot online", "deposit slot",
#     "slot pakai dana", "akun judi slot", "akun slot online"
# ]

# # Bangun bagian OR-nya
# or_clause = " OR ".join(f'"{phr}"' for phr in high_prec_phrases)

# QUERY = f"({or_clause}) lang:id -is:retweet -is:reply -filter:links -filter:media"

async def crawl_and_return(limit=500):
    config = ConfigParser()
    config.read('config.ini')
    username = config['X']['username']
    email = config['X']['email']
    password = config['X']['password']

    client = Client()
    client.load_cookies('cookies.json')

    tweet_count = 0
    tweets = None
    results = []

    high_prec_phrases = [
    "judi online", "main slot", "jackpot situs", "agen slot", "togel hari ini",
    "judol", "casino", "togel", "taruhan", "bola tangkas", "deposit judi slot", "cara daftar slot", "daftar slot", "slot online", "deposit slot",
    "slot pakai dana", "akun judi slot", "akun slot online"]

    # Bangun bagian OR-nya
    or_clause = " OR ".join(f'"{phr}"' for phr in high_prec_phrases)

    QUERY = f"({or_clause}) lang:id -is:retweet -is:reply -filter:links -filter:media"

    while tweet_count < limit:
        try:
            if tweets is None:
                tweets = await client.search_tweet(QUERY, product='Top')
            else:
                await asyncio.sleep(randint(3, 7))
                tweets = await tweets.next()

            if not tweets:
                break

            for tweet in tweets:
                tweet_count += 1

                # Tangani tanggal dengan aman
                created_raw = tweet.created_at
                if isinstance(created_raw, datetime):
                    created_at = created_raw.strftime("%Y-%m-%d %H:%M:%S")
                elif isinstance(created_raw, str):
                    try:
                        parsed = parser.parse(created_raw)  # Lebih fleksibel dari fromisoformat
                        created_at = parsed.strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        created_at = "Invalid Date"
                else:
                    created_at = "Unknown"

                results.append({
                    "text": tweet.text,
                    "user": tweet.user.name,
                    "created_at": created_at
                })

                if tweet_count >= limit:
                    break

        except TooManyRequests as e:
            wait_time = (datetime.fromtimestamp(e.rate_limit_reset) - datetime.now()).total_seconds()
            await asyncio.sleep(wait_time)

    return results
