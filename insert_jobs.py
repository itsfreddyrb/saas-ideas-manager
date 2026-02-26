import json
import os
import re
import time
import requests
import psycopg2
from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODEL = "minimax-m2.5:cloud"
OLLAMA_URL = "http://localhost:11434/api/generate"


def call_ollama(system_prompt, user_prompt, max_tokens=1024, retries=2):
    """Call Ollama API with the minimax model."""
    full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"
    
    for attempt in range(retries):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": full_prompt,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "num_predict": max_tokens
                    }
                },
                timeout=60
            )
            if response.status_code != 200:
                print(f"    Ollama API error: {response.status_code} - {response.text[:100]}")
                continue
                
            result = response.json()
            raw = result.get('response', '').strip()
            
            # Remove markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            
            # Try to parse JSON, if fails, try to extract JSON from text
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                # Try to find JSON in the response
                import re
                json_match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                print(f"    Failed to parse JSON, attempt {attempt + 1}/{retries}")
                continue
                
        except requests.exceptions.Timeout:
            print(f"    Ollama timeout, attempt {attempt + 1}/{retries}")
            continue
        except Exception as e:
            print(f"    Ollama error: {e}, attempt {attempt + 1}/{retries}")
            continue
    
    # Return a safe default if all retries fail
    return {"is_job": False, "reason": "LLM call failed"}


JOB_SOURCES = [
    'https://www.reddit.com/r/RemoteJobs/.json',
    'https://remotive.com/api/remote-jobs?category=software-dev&limit=25',
    'https://remoteok.com/api',
]

JOB_VALIDATION_SYSTEM_PROMPT = """You are an extremely strict filter. Your ONLY job is to determine if a post is a real, specific remote job listing in tech that someone could apply to RIGHT NOW.

To pass, the post MUST contain: a specific job title AND a company or employer AND enough detail to understand the role. It must be a tech role:
- Software engineering, programming, development
- Cloud engineering, DevOps, SRE, infrastructure
- Tech leadership, engineering management, CTO
- IT consulting, technical consulting
- Data engineering, data science, ML/AI engineering
- QA, security, database administration

REJECT everything else. When in doubt, REJECT. Specifically reject:
- Discussions, questions, advice, or personal stories of any kind
- Non-tech roles (sales, marketing, customer support, writing, design, data entry)
- Self-promotion, freelancer ads, "hire me" posts, or contractor platform pitches
- Surveys, meta posts, subreddit rules, or community threads
- Vague posts with just a job title and no details
- Aggregation posts like "jobs hiring this week" or "companies with remote roles"
- Gig work, beta testing, or "paid feedback" opportunities
- Posts about already having a job or job search experiences

Return ONLY a JSON object with exactly these fields:
- "is_job": true or false
- "reason": A short (1 sentence) explanation of your decision

Return ONLY the JSON object, no markdown fences, no extra text."""

JOB_ANALYSIS_SYSTEM_PROMPT = """You are a senior tech career advisor. You evaluate remote job postings for software engineers, cloud engineers, tech leads, and consultants.

Given a job posting, return a JSON object with exactly these fields:
- "summary": A concise 1-2 sentence summary of the role.
- "relevance_score": Integer 1-10. How relevant is this to tech/programming/cloud/consulting? (10 = core engineering role)
- "seniority_level": Exactly one of: "junior", "mid", "senior", "lead", "executive", "unknown"
- "skills": Key technical skills required (comma-separated string).
- "strengths": Key strengths of this opportunity (2-3 bullet points as a single string, separated by newlines).
- "weaknesses": Key weaknesses or red flags (2-3 bullet points as a single string, separated by newlines).
- "verdict": Exactly one of: "apply", "consider", or "skip".
- "llm_opinion": Your honest free-form opinion in 2-3 sentences. Would you recommend applying?

Return ONLY the JSON object, no markdown fences, no extra text."""


def strip_html(text):
    if not text:
        return text
    clean = re.sub(r'<[^>]+>', ' ', text)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean


def validate_job_with_llm(title, description):
    user_prompt = f"Title: {title}\nDescription: {description or 'No description provided'}"

    result = call_ollama(JOB_VALIDATION_SYSTEM_PROMPT, user_prompt, max_tokens=256)
    return result.get("is_job", False), result.get("reason", "")


def parse_reddit_jobs(data):
    jobs = []
    for post in data.get('data', {}).get('children', []):
        post_data = post.get('data', {})
        jobs.append({
            'title': post_data.get('title', ''),
            'company': None,
            'description': post_data.get('selftext', ''),
            'salary': None,
            'job_type': None,
            'location': 'Remote',
            'source': 'Reddit r/RemoteJobs',
            'url': f"https://reddit.com{post_data.get('permalink', '')}",
        })
    return jobs


def parse_remotive_jobs(data):
    jobs = []
    for job in data.get('jobs', [])[:25]:
        jobs.append({
            'title': job.get('title', ''),
            'company': job.get('company_name', ''),
            'description': strip_html(job.get('description', '')),
            'salary': job.get('salary', ''),
            'job_type': job.get('job_type', ''),
            'location': job.get('candidate_required_location', 'Remote'),
            'source': 'Remotive',
            'url': job.get('url', ''),
        })
    return jobs


def parse_remoteok_jobs(data):
    jobs = []
    for job in data[:25]:
        if not isinstance(job, dict) or 'id' not in job:
            continue
        tags = job.get('tags', [])
        tech_tags = {'dev', 'engineer', 'engineering', 'backend', 'frontend', 'fullstack',
                     'devops', 'cloud', 'python', 'javascript', 'golang', 'rust', 'java',
                     'react', 'node', 'aws', 'azure', 'gcp', 'kubernetes', 'docker',
                     'sre', 'infra', 'data', 'ml', 'ai', 'security', 'devsecops',
                     'software', 'senior', 'lead', 'architect', 'mobile', 'ios', 'android'}
        if tags and not any(t.lower() in tech_tags for t in tags):
            continue
        jobs.append({
            'title': job.get('position', ''),
            'company': job.get('company', ''),
            'description': strip_html(job.get('description', '')),
            'salary': f"{job.get('salary_min', '')}-{job.get('salary_max', '')}" if job.get('salary_min') else '',
            'job_type': None,
            'location': job.get('location', 'Remote'),
            'source': 'RemoteOK',
            'url': job.get('url', ''),
        })
    return jobs


def insert_job(cursor, job):
    cursor.execute("SELECT COUNT(*) FROM jobs WHERE title = %s", (job['title'],))
    if cursor.fetchone()[0] == 0:
        cursor.execute(
            "INSERT INTO jobs (title, company, description, salary, job_type, location, source, url) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (job['title'], job['company'], job['description'], job['salary'],
             job['job_type'], job['location'], job['source'], job['url'])
        )
        return True
    return False


def ensure_job_tables(conn):
    with conn.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id SERIAL PRIMARY KEY,
                title TEXT NOT NULL UNIQUE,
                company TEXT,
                description TEXT,
                salary TEXT,
                job_type TEXT,
                location TEXT,
                source TEXT,
                url TEXT,
                date_found TIMESTAMP DEFAULT NOW()
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS job_analysis (
                id SERIAL PRIMARY KEY,
                job_id INTEGER REFERENCES jobs(id) ON DELETE CASCADE,
                summary TEXT,
                relevance_score INTEGER,
                seniority_level TEXT,
                skills TEXT,
                strengths TEXT,
                weaknesses TEXT,
                verdict TEXT,
                llm_opinion TEXT,
                analyzed_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(job_id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_bookmarked_jobs (
                user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                job_id INTEGER REFERENCES jobs(id) ON DELETE CASCADE,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (user_id, job_id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_hidden_jobs (
                user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                job_id INTEGER REFERENCES jobs(id) ON DELETE CASCADE,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (user_id, job_id)
            )
        """)
    conn.commit()


def fetch_and_insert_jobs(conn):
    for source in JOB_SOURCES:
        try:
            response = requests.get(source, headers={'User-Agent': 'saas-ideas-bot/1.0'})
            if response.status_code != 200:
                print(f"Failed to fetch from {source}: {response.status_code}")
                continue

            jobs = []
            if 'reddit.com' in source:
                jobs = parse_reddit_jobs(response.json())
            elif 'remotive.com' in source:
                jobs = parse_remotive_jobs(response.json())
            elif 'remoteok.com' in source:
                jobs = parse_remoteok_jobs(response.json())

            with conn.cursor() as cursor:
                inserted = 0
                skipped = 0
                for i, job in enumerate(jobs):
                    if i > 0:
                        time.sleep(1.5)
                    try:
                        is_job, reason = validate_job_with_llm(job['title'], job['description'])
                    except Exception as e:
                        print(f"  Validation error for '{job['title'][:60]}': {e} — skipping")
                        skipped += 1
                        continue

                    if not is_job:
                        print(f"  Rejected: {job['title'][:60]} — {reason}")
                        skipped += 1
                        continue

                    print(f"  Accepted: {job['title'][:60]}")
                    if insert_job(cursor, job):
                        inserted += 1
            conn.commit()
            print(f"Fetched {len(jobs)} from {source}, accepted {len(jobs) - skipped}, inserted {inserted} new")
        except Exception as e:
            conn.rollback()
            print(f"Error fetching from {source}: {e}")


def analyze_job_with_llm(job_row):
    job_id, title, company, description, salary, job_type, location, source, url = job_row

    user_prompt = f"""Evaluate this remote job posting:

Title: {title}
Company: {company or 'Unknown'}
Description: {description or 'No description provided'}
Salary: {salary or 'Not specified'}
Type: {job_type or 'Unknown'}
Location: {location or 'Remote'}
Source: {source or 'Unknown'}"""

    result = call_ollama(JOB_ANALYSIS_SYSTEM_PROMPT, user_prompt, max_tokens=1024)
    return result


def analyze_jobs(conn):
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT j.id, j.title, j.company, j.description, j.salary,
                   j.job_type, j.location, j.source, j.url
            FROM jobs j
            LEFT JOIN job_analysis a ON j.id = a.job_id
            WHERE a.id IS NULL
        """)
        unanalyzed = cursor.fetchall()

    if not unanalyzed:
        print("All jobs have already been analyzed.")
        return

    print(f"\nAnalyzing {len(unanalyzed)} jobs with Ollama ({OLLAMA_MODEL})...")

    for i, job_row in enumerate(unanalyzed):
        if i > 0:
            time.sleep(1.5)
        job_id = job_row[0]
        title = job_row[1]
        try:
            analysis = analyze_job_with_llm(job_row)
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO job_analysis "
                    "(job_id, summary, relevance_score, seniority_level, skills, "
                    "strengths, weaknesses, verdict, llm_opinion) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (job_id, analysis['summary'], analysis['relevance_score'],
                     analysis['seniority_level'], analysis['skills'],
                     analysis['strengths'], analysis['weaknesses'],
                     analysis['verdict'], analysis['llm_opinion'])
                )
            conn.commit()
            print(f"  Analyzed: {title[:60]} -> {analysis['verdict']} (relevance: {analysis['relevance_score']}/10)")
        except Exception as e:
            conn.rollback()
            print(f"  Error analyzing '{title[:60]}': {e}")


if __name__ == '__main__':
    conn = psycopg2.connect(
        dbname=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        host=os.getenv('DB_HOST'),
        password=os.getenv('DB_PASSWORD'),
        port=os.getenv('DB_PORT', 5432),
    )

    ensure_job_tables(conn)
    fetch_and_insert_jobs(conn)
    analyze_jobs(conn)
    conn.close()
