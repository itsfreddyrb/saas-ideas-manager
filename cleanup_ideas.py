import json
import os
import time
import psycopg2
import anthropic
from dotenv import load_dotenv

load_dotenv()

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


def validate_idea_with_llm(client, title, description):
    user_prompt = f"Title: {title}\nDescription: {description or 'No description provided'}"

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system=VALIDATION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    result = json.loads(raw)
    return result.get("is_idea", False), result.get("reason", "")


if __name__ == '__main__':
    conn = psycopg2.connect(
        dbname=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        host=os.getenv('DB_HOST'),
        password=os.getenv('DB_PASSWORD'),
        port=os.getenv('DB_PORT', 5432),
    )
    client = anthropic.Anthropic()

    with conn.cursor() as cursor:
        cursor.execute("SELECT id, idea, description FROM ideas ORDER BY id")
        all_ideas = cursor.fetchall()

    print(f"Found {len(all_ideas)} ideas in the database. Validating...\n")

    to_delete = []
    for i, (idea_id, title, description) in enumerate(all_ideas):
        if i > 0:
            time.sleep(1.5)
        try:
            is_idea, reason = validate_idea_with_llm(client, title, description)
        except Exception as e:
            print(f"  [ERROR] id={idea_id} '{title[:60]}': {e} — keeping")
            continue

        if not is_idea:
            print(f"  [DELETE] id={idea_id} '{title[:60]}' — {reason}")
            to_delete.append(idea_id)
        else:
            print(f"  [KEEP]   id={idea_id} '{title[:60]}'")

    print(f"\n--- Summary ---")
    print(f"Total: {len(all_ideas)} | Keep: {len(all_ideas) - len(to_delete)} | Delete: {len(to_delete)}")

    if to_delete:
        confirm = input(f"\nDelete {len(to_delete)} non-ideas? (yes/no): ").strip().lower()
        if confirm == 'yes':
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM ideas WHERE id = ANY(%s)", (to_delete,))
            conn.commit()
            print(f"Deleted {len(to_delete)} rows (and their analyses via CASCADE).")
        else:
            print("Aborted. No rows deleted.")
    else:
        print("Nothing to delete.")

    conn.close()
