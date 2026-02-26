import json
import os
import re
import time
import xml.etree.ElementTree as ET
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

# ─────────────────────────────────────────────
# Sources — intentionally different from insert_jobs.py
# (which already covers Reddit r/RemoteJobs, Remotive, RemoteOK)
# ─────────────────────────────────────────────
JOB_SOURCES = [
    {
        'url': 'https://weworkremotely.com/categories/remote-programming-jobs.rss',
        'type': 'rss',
        'name': 'We Work Remotely',
    },
    {
        'url': 'https://www.arbeitnow.com/api/job-board-api',
        'type': 'arbeitnow',
        'name': 'Arbeitnow',
    },
    {
        'url': 'https://jobicy.com/api/v2/remote-jobs?count=25&industry=technology',
        'type': 'jobicy',
        'name': 'Jobicy',
    },
]

# ─────────────────────────────────────────────
# LLM System Prompts
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def strip_html(text):
    """Remove HTML tags and collapse whitespace."""
    if not text:
        return text
    clean = re.sub(r'<[^>]+>', ' ', text)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean


def safe_get(url, name, timeout=15):
    """HTTP GET with a descriptive error on failure."""
    headers = {'User-Agent': 'saas-ideas-scraper/2.0'}
    response = requests.get(url, headers=headers, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(f"[{name}] HTTP {response.status_code} for {url}")
    return response


# ─────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────
def parse_wwr_rss(xml_text):
    """
    Parse We Work Remotely RSS feed.
    Titles come in the form "CompanyName: Job Title", so we split on the first colon.
    """
    jobs = []
    root = ET.fromstring(xml_text)
    ns = {'content': 'http://purl.org/rss/1.0/modules/content/'}

    for item in root.iter('item'):
        raw_title = (item.findtext('title') or '').strip()
        link      = (item.findtext('link') or '').strip()
        desc_html = item.findtext('description') or ''
        content   = item.find('content:encoded', ns)
        body      = strip_html(content.text if content is not None else desc_html)

        # "Company: Role" → split company from role
        if ':' in raw_title:
            company, title = raw_title.split(':', 1)
            company = company.strip()
            title   = title.strip()
        else:
            company = None
            title   = raw_title

        if not title:
            continue

        jobs.append({
            'title':       title,
            'company':     company,
            'description': body[:2000],
            'salary':      None,
            'job_type':    'full-time',
            'location':    'Remote',
            'source':      'We Work Remotely',
            'url':         link,
        })
    return jobs


def parse_arbeitnow(data):
    """
    Parse Arbeitnow API response.
    Docs: https://www.arbeitnow.com/api/job-board-api
    Only keep remote-flagged listings.
    """
    jobs = []
    for job in data.get('data', [])[:25]:
        if not job.get('remote', False):
            continue
        tags = job.get('tags', [])
        tech_keywords = {
            'developer', 'engineer', 'engineering', 'backend', 'frontend',
            'fullstack', 'devops', 'cloud', 'python', 'javascript', 'golang',
            'rust', 'java', 'react', 'node', 'aws', 'azure', 'gcp',
            'kubernetes', 'docker', 'sre', 'infrastructure', 'data', 'ml',
            'ai', 'security', 'mobile', 'ios', 'android', 'architect', 'lead',
        }
        title_lower = job.get('title', '').lower()
        tag_lower   = ' '.join(tags).lower()
        if not any(kw in title_lower or kw in tag_lower for kw in tech_keywords):
            continue

        jobs.append({
            'title':       job.get('title', ''),
            'company':     job.get('company_name', ''),
            'description': strip_html(job.get('description', ''))[:2000],
            'salary':      None,
            'job_type':    ', '.join(job.get('job_types', [])) or None,
            'location':    job.get('location', 'Remote'),
            'source':      'Arbeitnow',
            'url':         job.get('url', ''),
        })
    return jobs


def parse_jobicy(data):
    """
    Parse Jobicy API response.
    Docs: https://jobicy.com/jobs-rss-feed
    """
    jobs = []
    for job in data.get('jobs', [])[:25]:
        jobs.append({
            'title':       job.get('jobTitle', ''),
            'company':     job.get('companyName', ''),
            'description': strip_html(job.get('jobDescription', job.get('jobExcerpt', '')))[:2000],
            'salary':      None,
            'job_type':    ', '.join(job.get('jobType', [])) if isinstance(job.get('jobType'), list) else job.get('jobType'),
            'location':    job.get('jobGeo', 'Remote'),
            'source':      'Jobicy',
            'url':         job.get('url', ''),
        })
    return jobs


# ─────────────────────────────────────────────
# LLM Calls
# ─────────────────────────────────────────────
def validate_job_with_llm(title, description):
    user_prompt = f"Title: {title}\nDescription: {description or 'No description provided'}"

    result = call_ollama(JOB_VALIDATION_SYSTEM_PROMPT, user_prompt, max_tokens=256)
    return result.get("is_job", False), result.get("reason", "")


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


# ─────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────
def ensure_tables(conn):
    """Create jobs and job_analysis tables if they don't already exist."""
    with conn.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id          SERIAL PRIMARY KEY,
                title       TEXT NOT NULL,
                company     TEXT,
                description TEXT,
                salary      TEXT,
                job_type    TEXT,
                location    TEXT,
                source      TEXT,
                url         TEXT,
                date_found  TIMESTAMP DEFAULT NOW(),
                UNIQUE(title, company)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS job_analysis (
                id               SERIAL PRIMARY KEY,
                job_id           INTEGER REFERENCES jobs(id) ON DELETE CASCADE,
                summary          TEXT,
                relevance_score  INTEGER,
                seniority_level  TEXT,
                skills           TEXT,
                strengths        TEXT,
                weaknesses       TEXT,
                verdict          TEXT,
                llm_opinion      TEXT,
                analyzed_at      TIMESTAMP DEFAULT NOW(),
                UNIQUE(job_id)
            )
        """)
    conn.commit()


def insert_job(cursor, job):
    """Insert a job if it doesn't already exist. Returns True if a new row was inserted."""
    cursor.execute(
        "SELECT COUNT(*) FROM jobs WHERE title = %s AND company IS NOT DISTINCT FROM %s",
        (job['title'], job['company']),
    )
    if cursor.fetchone()[0] > 0:
        return False

    cursor.execute(
        """
        INSERT INTO jobs (title, company, description, salary, job_type, location, source, url)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            job['title'], job['company'], job['description'], job['salary'],
            job['job_type'], job['location'], job['source'], job['url'],
        ),
    )
    return True


# ─────────────────────────────────────────────
# Pipeline Steps
# ─────────────────────────────────────────────
def fetch_and_insert_jobs(conn):
    """Fetch jobs from all sources, validate with LLM, insert accepted ones."""

    for source in JOB_SOURCES:
        name = source['name']
        url  = source['url']
        kind = source['type']

        print(f"\n{'─'*55}")
        print(f"  Source: {name}")
        print(f"{'─'*55}")

        try:
            response = safe_get(url, name)
        except Exception as e:
            print(f"  [FETCH ERROR] {e}")
            continue

        # Parse according to source type
        try:
            if kind == 'rss':
                jobs = parse_wwr_rss(response.text)
            elif kind == 'arbeitnow':
                jobs = parse_arbeitnow(response.json())
            elif kind == 'jobicy':
                jobs = parse_jobicy(response.json())
            else:
                print(f"  [SKIP] Unknown source type: {kind}")
                continue
        except Exception as e:
            print(f"  [PARSE ERROR] {e}")
            continue

        print(f"  Fetched {len(jobs)} candidate(s) from {name}")

        inserted = 0
        rejected = 0

        with conn.cursor() as cursor:
            for i, job in enumerate(jobs):
                if not job.get('title'):
                    rejected += 1
                    continue

                # Rate-limit LLM calls
                if i > 0:
                    time.sleep(1.5)

                # LLM validation gate
                try:
                    is_job, reason = validate_job_with_llm(job['title'], job['description'])
                except Exception as e:
                    print(f"  [VALIDATION ERROR] '{job['title'][:60]}': {e} — skipping")
                    rejected += 1
                    continue

                if not is_job:
                    print(f"  [REJECTED] {job['title'][:60]} — {reason}")
                    rejected += 1
                    continue

                print(f"  [ACCEPTED] {job['title'][:60]}")

                try:
                    if insert_job(cursor, job):
                        inserted += 1
                    else:
                        print(f"             ↳ already in DB, skipped")
                except Exception as e:
                    print(f"  [INSERT ERROR] '{job['title'][:60]}': {e}")
                    rejected += 1

        try:
            conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"  [COMMIT ERROR] {name}: {e}")
            continue

        accepted = len(jobs) - rejected
        print(f"\n  Summary → fetched: {len(jobs)} | accepted: {accepted} | inserted new: {inserted}")


def analyze_jobs(conn):
    """Run LLM analysis on every job that doesn't have an analysis row yet."""

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
        print("\n  All jobs already have analysis. Nothing to do.")
        return

    print(f"\n{'─'*55}")
    print(f"  Analyzing {len(unanalyzed)} unanalyzed job(s) with Ollama ({OLLAMA_MODEL})…")
    print(f"{'─'*55}")

    for i, job_row in enumerate(unanalyzed):
        if i > 0:
            time.sleep(1.5)

        job_id = job_row[0]
        title  = job_row[1]

        try:
            analysis = analyze_job_with_llm(job_row)
        except Exception as e:
            print(f"  [ANALYSIS ERROR] '{title[:60]}': {e}")
            continue

        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO job_analysis
                        (job_id, summary, relevance_score, seniority_level, skills,
                         strengths, weaknesses, verdict, llm_opinion)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (job_id) DO NOTHING
                    """,
                    (
                        job_id,
                        analysis.get('summary'),
                        analysis.get('relevance_score'),
                        analysis.get('seniority_level'),
                        analysis.get('skills'),
                        analysis.get('strengths'),
                        analysis.get('weaknesses'),
                        analysis.get('verdict'),
                        analysis.get('llm_opinion'),
                    ),
                )
            conn.commit()
            verdict   = analysis.get('verdict', '?')
            relevance = analysis.get('relevance_score', '?')
            print(f"  [ANALYZED] {title[:60]} → {verdict} (relevance: {relevance}/10)")
        except Exception as e:
            conn.rollback()
            print(f"  [SAVE ERROR] '{title[:60]}': {e}")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == '__main__':
    conn = psycopg2.connect(
        dbname=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        host=os.getenv('DB_HOST'),
        password=os.getenv('DB_PASSWORD'),
        port=os.getenv('DB_PORT', 5432),
    )

    try:
        ensure_tables(conn)
        fetch_and_insert_jobs(conn)
        analyze_jobs(conn)
    finally:
        conn.close()
        print("\n  Done. Connection closed.")
