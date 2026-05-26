import os
import re
import sys
import logging
import requests
import feedparser
from flask import Flask, jsonify, request
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configurable Tech RSS Feeds (Greenhouse and Lever endpoints as targets)
JOB_FEEDS = [
    "https://boards.greenhouse.io/v1/boards/cohere/jobs?rss=true",
    "https://boards.greenhouse.io/v1/boards/openai/jobs?rss=true",
    "https://jobs.lever.co/google/rss",
    "https://jobs.lever.co/lever/rss"
]

# Pydantic models for structured AI engine responses
class JobEvaluation(BaseModel):
    index: int = Field(description="The index of the job in the batch list")
    category: str = Field(
        description="Must be exactly one of: 'Software Engineering', 'AI/ML Engineering', 'Cybersecurity Engineering', or 'Other'"
    )
    score: int = Field(
        description="A technology syntax matching score from 0 to 100 based on alignment with the candidate's resume"
    )
    reasoning: str = Field(
        description="A precise one-sentence explanation for the score based on skills, tech stack, and experience alignment."
    )

class JobEvaluationBatch(BaseModel):
    evaluations: List[JobEvaluation]

def get_resume_content() -> str:
    """Reads resume.md from the workspace under UTF-8 encoding rules."""
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    resume_path = os.path.join(workspace_dir, "resume.md")
    
    if not os.path.exists(resume_path):
        logger.error(f"resume.md not found at path: {resume_path}")
        raise FileNotFoundError("resume.md is missing from the workspace root")
        
    with open(resume_path, "r", encoding="utf-8") as f:
        return f.read()

def fetch_rss_jobs() -> List[Dict[str, Any]]:
    """Streams and parses tech jobs from targeted feeds, normalizing the payload."""
    raw_jobs = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    for feed_url in RSS_FEEDS:
        try:
            logger.info(f"Fetching RSS feed from: {feed_url}")
            response = requests.get(feed_url, headers=headers, timeout=15)
            response.raise_for_status()
            
            feed = feedparser.parse(response.text)
            
            # Determine company name from the feed title or URL structure
            feed_title = feed.feed.get("title", "")
            company = "Unknown"
            if "Greenhouse" in feed_title or "greenhouse" in feed_url:
                # E.g. "OpenAI Jobs" -> "OpenAI"
                company = feed_title.replace("Jobs", "").strip()
            elif "Lever" in feed_title or "lever" in feed_url:
                # E.g. "Lever Jobs RSS Feed" -> "Lever"
                company = feed_title.split(" ")[0].strip()
            
            for entry in feed.entries:
                title = entry.get("title", "Unknown Title")
                link = entry.get("link", "")
                
                # Extract and sanitize description
                description_html = entry.get("summary", entry.get("description", ""))
                description_clean = re.sub(r"<[^<]+?>", "", description_html)
                # Replace multiple whitespaces/newlines with a single space
                description_clean = re.sub(r"\s+", " ", description_clean).strip()
                description_capped = (
                    description_clean[:400] + "..." if len(description_clean) > 400 else description_clean
                )
                
                raw_jobs.append({
                    "title": title,
                    "company": company,
                    "link": link,
                    "description": description_capped
                })
                
        except Exception as e:
            logger.error(f"Error processing RSS feed {feed_url}: {e}", exc_info=True)
            
    logger.info(f"Successfully aggregated {len(raw_jobs)} job postings.")
    return raw_jobs

def evaluate_jobs_with_ai(jobs: List[Dict[str, Any]], resume_text: str) -> List[Dict[str, Any]]:
    """Invokes gemini-2.5-flash with isolated configurations using the native google-genai SDK."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable is not set.")
        raise ValueError("GEMINI_API_KEY is not configured")
        
    client = genai.Client(api_key=api_key)
    
    # Format jobs context for the AI prompt
    jobs_formatted = []
    for idx, job in enumerate(jobs):
        jobs_formatted.append(
            f"Index: {idx}\nTitle: {job['title']}\nCompany: {job['company']}\nDescription: {job['description']}\n---"
        )
    jobs_context = "\n".join(jobs_formatted)
    
    system_instruction = (
        "You are an elite, deterministic AI recruiter and taxonomy gate. Your job is to strictly classify and score a list of job postings against a candidate's resume.\n\n"
        "1. Classification:\n"
        "You must classify each job into one of the following exact spelling-precise categories:\n"
        "- 'Software Engineering'\n"
        "- 'AI/ML Engineering'\n"
        "- 'Cybersecurity Engineering'\n"
        "If a job does not fit perfectly into one of these, classify it as 'Other'.\n\n"
        "2. Scoring:\n"
        "Evaluate the job on a 0-100 index based on exact technology syntax and skill alignment with the candidate's resume.\n\n"
        "3. Telegram Formatting Constraint:\n"
        "Avoid using double asterisks (**bold**) under any circumstances in your reasoning or fields. "
        "Strictly utilize single asterisks (*bold*) for styling headers and labels to align flawlessly with Telegram's Markdown parsing logic and avoid rendering raw escape characters."
    )
    
    prompt = (
        f"Candidate Resume:\n{resume_text}\n\n"
        f"Jobs list to analyze:\n{jobs_context}\n\n"
        "Please evaluate all jobs and return the classifications, scores, and brief reasonings."
    )
    
    try:
        logger.info(f"Sending {len(jobs)} jobs to Gemini model for evaluation...")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=JobEvaluationBatch,
                temperature=0.0
            )
        )
        
        # Parse the structured response
        evaluations_batch = response.parsed
        evaluations_map = {eval_item.index: eval_item for eval_item in evaluations_batch.evaluations}
        
        evaluated_jobs = []
        for idx, job in enumerate(jobs):
            eval_data = evaluations_map.get(idx)
            if eval_data:
                # Merge AI evaluation into original job structure
                evaluated_jobs.append({
                    **job,
                    "category": eval_data.category,
                    "score": eval_data.score,
                    "reasoning": eval_data.reasoning
                })
            else:
                logger.warning(f"No evaluation returned for job index {idx}")
                
        return evaluated_jobs
        
    except Exception as e:
        logger.error(f"Error during AI job evaluation: {e}", exc_info=True)
        raise

def process_and_filter_jobs(evaluated_jobs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Filters, sorts, and caps jobs per spelling-precise category in Python for deterministic correctness."""
    allowed_categories = ["Software Engineering", "AI/ML Engineering", "Cybersecurity Engineering"]
    
    # Filter: Score must be >= 75 and category must be strictly valid
    filtered = [
        job for job in evaluated_jobs 
        if job.get("category") in allowed_categories and job.get("score", 0) >= 75
    ]
    
    # Group and sort by score descending
    categorized_jobs: Dict[str, List[Dict[str, Any]]] = {cat: [] for cat in allowed_categories}
    for job in filtered:
        categorized_jobs[job["category"]].append(job)
        
    # Sort and cap to top 5 per category
    total_processed = 0
    final_structured_jobs = {}
    for cat in allowed_categories:
        sorted_cat_jobs = sorted(categorized_jobs[cat], key=lambda x: x["score"], reverse=True)
        capped_cat_jobs = sorted_cat_jobs[:5]
        final_structured_jobs[cat] = capped_cat_jobs
        total_processed += len(capped_cat_jobs)
        logger.info(f"Category '{cat}': found {len(capped_cat_jobs)} qualifying jobs (after capping).")
        
    return final_structured_jobs

def build_telegram_markdown(structured_jobs: Dict[str, List[Dict[str, Any]]]) -> str:
    """Builds a beautifully styled Telegram message strictly using single asterisks (*bold*) instead of double asterisks."""
    has_jobs = any(len(jobs) > 0 for jobs in structured_jobs.values())
    if not has_jobs:
        return "*Job Alert Intelligence Report*\n\nNo matching jobs matching the score criteria (>=75) were found in this fetch cycle."
        
    markdown_lines = ["*Job Alert Intelligence Report*\n"]
    
    for category, jobs in structured_jobs.items():
        if not jobs:
            continue
        markdown_lines.append(f"*{category}*")
        for job in jobs:
            title = job["title"]
            company = job["company"]
            link = job["link"]
            score = job["score"]
            reasoning = job["reasoning"]
            
            # Format using strictly single asterisks for bolding to align with Telegram MarkdownV1 or MarkdownV2 escaping
            markdown_lines.append(f"• *{company}* - *{title}* (Score: *{score}*)")
            markdown_lines.append(f"  Link: {link}")
            markdown_lines.append(f"  Reasoning: {reasoning}")
        markdown_lines.append("") # Empty line separator
        
    return "\n".join(markdown_lines).strip()

def send_telegram_notifications(message: str) -> None:
    """Sends the formatted markdown to the Telegram bot endpoint using safe length chunking."""
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        logger.warning("Telegram configuration is incomplete. Skipping message broadcast.")
        return
        
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    
    # Safe chunking to fit the 4000 character limit
    max_length = 4000
    if len(message) <= max_length:
        chunks = [message]
    else:
        chunks = []
        current_chunk = ""
        for line in message.split("\n"):
            if len(current_chunk) + len(line) + 1 > max_length:
                chunks.append(current_chunk.strip())
                current_chunk = line + "\n"
            else:
                current_chunk += line + "\n"
        if current_chunk:
            chunks.append(current_chunk.strip())
            
    for idx, chunk in enumerate(chunks):
        payload = {
            "chat_id": chat_id,
            "text": chunk,
            "parse_mode": "Markdown"
        }
        try:
            logger.info(f"Broadcasting message chunk {idx + 1}/{len(chunks)} to Telegram...")
            response = requests.post(url, json=payload, timeout=15)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to send Telegram message chunk {idx + 1}: {e}", exc_info=True)
            raise

@app.route("/fetch-jobs", methods=["GET", "POST"])
def fetch_jobs():
    """Headless automated orchestration endpoint for job processing pipeline."""
    try:
        # Context Processing: verify and read resume.md
        try:
            resume_text = get_resume_content()
        except FileNotFoundError as fnf_err:
            return jsonify({"error": str(fnf_err)}), 404
            
        # Streaming Aggregator: fetch and parse RSS feeds
        raw_jobs = fetch_rss_jobs()
        if not raw_jobs:
            return jsonify({"status": "success", "jobs_processed": 0})
            
        # AI Studio Engine: Classify, score, and reason
        evaluated_jobs = evaluate_jobs_with_ai(raw_jobs, resume_text)
        
        # Classification & Hard Filtering: process scoring and category matching
        structured_jobs = process_and_filter_jobs(evaluated_jobs)
        
        # Count total jobs formatted and broadcasted
        total_jobs_broadcasted = sum(len(jobs) for jobs in structured_jobs.values())
        
        # Build Markdown Payload
        telegram_message = build_telegram_markdown(structured_jobs)
        
        # Resilient Broadcast Pipeline
        send_telegram_notifications(telegram_message)
        
        # Clean Cron Payload Fix: Return a lean JSON response to prevent timer crashes
        return jsonify({
            "status": "success",
            "jobs_processed": total_jobs_broadcasted
        })
        
    except Exception as err:
        logger.error(f"Pipeline error: {err}", exc_info=True)
        return jsonify({"error": "Internal pipeline execution failure", "details": str(err)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    # Production-grade defaults when executed directly
    app.run(host="0.0.0.0", port=port, debug=False)
