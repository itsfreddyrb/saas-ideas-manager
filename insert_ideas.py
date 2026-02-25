import json
import os
import csv
import time
import requests
import psycopg2
import anthropic
from dotenv import load_dotenv

load_dotenv()


SOURCES = [
    'https://www.reddit.com/r/SaaS/.json',
    'https://www.reddit.com/r/entrepreneur/.json',
    'https://hacker-news.firebaseio.com/v0/newstories.json',
]

REDDIT_SOURCE_NAMES = {
    'https://www.reddit.com/r/SaaS/.json': 'Reddit r/SaaS',
    'https://www.reddit.com/r/entrepreneur/.json': 'Reddit r/entrepreneur',
}

VALIDATION_SYSTEM_PROMPT = """You are an extremely strict filter. Your ONLY job is to determine if a post contains a specific, concrete SaaS or software product idea that someone could build.

To pass, the post MUST explicitly describe a specific product, tool, or software service — including what it does, who it's for, or what problem it solves. The idea must be clear enough that a developer could start building it.

REJECT everything else. When in doubt, REJECT. Specifically reject:
- Personal stories, journeys, or "how I built X" retrospectives
- Success stories, case studies, or revenue milestone posts
- Questions, advice requests, or discussions of any kind
- Rants, opinions, hot takes, or market commentary
- Job postings, promotions, self-promotion, or "check out my tool" posts
- Surveys, polls, meta posts, or community threads
- Vague statements like "I want to build a SaaS" without describing what it does
- Posts about existing/already-launched products (these are NOT ideas)
- Aggregation posts like "here are some ideas" or "top picks this week"

Return ONLY a JSON object with exactly these fields:
- "is_idea": true or false
- "reason": A short (1 sentence) explanation of your decision

Return ONLY the JSON object, no markdown fences, no extra text."""

ANALYSIS_SYSTEM_PROMPT = """You are a seasoned SaaS startup advisor. You evaluate SaaS product ideas for solo developers and small teams.

Given a SaaS idea, return a JSON object with exactly these fields:
- "summary": A concise 1-2 sentence summary of the idea.
- "feasibility_score": Integer 1-10. How realistic is this to build? (10 = very easy to build)
- "market_potential_score": Integer 1-10. How large is the market opportunity? (10 = massive demand)
- "effort_score": Integer 1-10. How easy is it to ship an MVP? (10 = weekend project, 1 = months of work)
- "overall_score": Integer 1-10. Your overall recommendation combining all factors.
- "monetization_suggestion": How should this be monetized? Be specific.
- "strengths": Key strengths of this idea (2-3 bullet points as a single string, separated by newlines).
- "weaknesses": Key weaknesses or risks (2-3 bullet points as a single string, separated by newlines).
- "verdict": Exactly one of: "build", "consider", or "discard".
- "llm_opinion": Your honest free-form opinion in 2-3 sentences. Be direct — would you personally invest time in this?

Return ONLY the JSON object, no markdown fences, no extra text."""


def validate_idea_with_llm(client, title, description):
    user_prompt = f"Title: {title}\nDescription: {description or 'No description provided'}"

    message = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=256,
        system=[{"type": "text", "text": VALIDATION_SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    result = json.loads(raw)
    return result.get("is_idea", False), result.get("reason", "")


def parse_reddit_response(data, source_name='Reddit r/SaaS'):
    ideas = []
    for post in data.get('data', {}).get('children', []):
        post_data = post.get('data', {})
        ideas.append({
            'title': post_data.get('title', ''),
            'description': post_data.get('selftext', ''),
            'difficulty': None,
            'effort_est': None,
            'monetization': None,
            'source': source_name,
            'date_found': None,
            'notes': None,
        })
    return ideas


def parse_hacker_news_response(data):
    ideas = []
    for story_id in data[:10]:
        story = requests.get(f'https://hacker-news.firebaseio.com/v0/item/{story_id}.json').json()
        if story:
            ideas.append({
                'title': story.get('title', ''),
                'description': story.get('url', ''),
                'difficulty': None,
                'effort_est': None,
                'monetization': None,
                'source': 'Hacker News',
                'date_found': None,
                'notes': None,
            })
    return ideas


def insert_idea(cursor, idea):
    cursor.execute("SELECT COUNT(*) FROM ideas WHERE idea = %s", (idea['title'],))
    if cursor.fetchone()[0] == 0:
        cursor.execute(
            "INSERT INTO ideas (idea, description, difficulty, effort_est, monetization, source, date_found, notes) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (idea['title'], idea['description'], idea['difficulty'], idea['effort_est'],
             idea['monetization'], idea['source'], idea['date_found'], idea['notes'])
        )
        return True
    return False


def fetch_and_insert_ideas(conn):
    client = anthropic.Anthropic()

    for source in SOURCES:
        try:
            response = requests.get(source, headers={'User-Agent': 'saas-ideas-bot/1.0'})
            if response.status_code != 200:
                print(f"Failed to fetch from {source}: {response.status_code}")
                continue

            ideas = []
            if source in REDDIT_SOURCE_NAMES:
                ideas = parse_reddit_response(response.json(), REDDIT_SOURCE_NAMES[source])
            elif source == 'https://hacker-news.firebaseio.com/v0/newstories.json':
                ideas = parse_hacker_news_response(response.json())

            with conn.cursor() as cursor:
                inserted = 0
                skipped_validation = 0
                for i, idea in enumerate(ideas):
                    if i > 0:
                        time.sleep(1.5)
                    try:
                        is_idea, reason = validate_idea_with_llm(client, idea['title'], idea['description'])
                    except Exception as e:
                        print(f"  Validation error for '{idea['title'][:60]}': {e} — skipping")
                        skipped_validation += 1
                        continue

                    if not is_idea:
                        print(f"  Rejected: {idea['title'][:60]} — {reason}")
                        skipped_validation += 1
                        continue

                    print(f"  Accepted: {idea['title'][:60]}")
                    if insert_idea(cursor, idea):
                        inserted += 1
            conn.commit()
            print(f"Fetched {len(ideas)} from {source}, accepted {len(ideas) - skipped_validation}, inserted {inserted} new")
        except Exception as e:
            conn.rollback()
            print(f"Error fetching from {source}: {e}")


def ensure_analysis_table(conn):
    with conn.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS idea_analysis (
                id SERIAL PRIMARY KEY,
                idea_id INTEGER REFERENCES ideas(id) ON DELETE CASCADE,
                summary TEXT,
                feasibility_score INTEGER,
                market_potential_score INTEGER,
                effort_score INTEGER,
                overall_score INTEGER,
                monetization_suggestion TEXT,
                strengths TEXT,
                weaknesses TEXT,
                verdict TEXT,
                llm_opinion TEXT,
                analyzed_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(idea_id)
            )
        """)
    conn.commit()


def analyze_idea_with_llm(client, idea_row):
    idea_id, title, description, difficulty, effort_est, monetization, source = idea_row

    user_prompt = f"""Evaluate this SaaS idea:

Title: {title}
Description: {description or 'No description provided'}
Difficulty: {difficulty or 'Unknown'}
Estimated effort (hours): {effort_est or 'Unknown'}
Current monetization idea: {monetization or 'None'}
Source: {source or 'Unknown'}"""

    message = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        system=[{"type": "text", "text": ANALYSIS_SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_prompt}],
    )

    if not message.content:
        raise ValueError(f"Empty response from API (stop_reason: {message.stop_reason})")
    raw = message.content[0].text.strip()
    if not raw:
        raise ValueError(f"Empty text in response (stop_reason: {message.stop_reason})")
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(raw)


def analyze_ideas(conn):
    client = anthropic.Anthropic()

    ensure_analysis_table(conn)

    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT i.id, i.idea, i.description, i.difficulty, i.effort_est, i.monetization, i.source
            FROM ideas i
            LEFT JOIN idea_analysis a ON i.id = a.idea_id
            WHERE a.id IS NULL
        """)
        unanalyzed = cursor.fetchall()

    if not unanalyzed:
        print("All ideas have already been analyzed.")
        return

    print(f"\nAnalyzing {len(unanalyzed)} ideas with Claude...")

    for i, idea_row in enumerate(unanalyzed):
        if i > 0:
            time.sleep(1.5)
        idea_id = idea_row[0]
        title = idea_row[1]
        try:
            analysis = analyze_idea_with_llm(client, idea_row)
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO idea_analysis "
                    "(idea_id, summary, feasibility_score, market_potential_score, effort_score, "
                    "overall_score, monetization_suggestion, strengths, weaknesses, verdict, llm_opinion) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (idea_id, analysis['summary'], analysis['feasibility_score'],
                     analysis['market_potential_score'], analysis['effort_score'],
                     analysis['overall_score'], analysis['monetization_suggestion'],
                     analysis['strengths'], analysis['weaknesses'],
                     analysis['verdict'], analysis['llm_opinion'])
                )
            conn.commit()
            print(f"  Analyzed: {title[:60]} -> {analysis['verdict']} (score: {analysis['overall_score']}/10)")
        except Exception as e:
            conn.rollback()
            print(f"  Error analyzing '{title[:60]}': {e}")



def insert_ideas_from_csv(conn, csv_path):
    with open(csv_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header
        with conn.cursor() as cursor:
            for row in csv_reader:
                idea = {
                    'title': row[1],
                    'description': row[2],
                    'difficulty': row[3],
                    'effort_est': row[4],
                    'monetization': row[5],
                    'source': row[6],
                    'date_found': row[7],
                    'notes': row[8] if len(row) > 8 else '',
                }
                insert_idea(cursor, idea)
        conn.commit()


if __name__ == '__main__':
    conn = psycopg2.connect(
        dbname=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        host=os.getenv('DB_HOST'),
        password=os.getenv('DB_PASSWORD'),
        port=os.getenv('DB_PORT', 5432),
    )

    fetch_and_insert_ideas(conn)
    analyze_ideas(conn)
    conn.close()
