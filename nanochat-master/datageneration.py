"""
Synthetic data generator to imprint a Thronglet-style identity.
Now with checkpoint/resume capability - restarts pick up where they left off.

WHAT CHANGED vs the original:
- Identity is now a digital lifeform called "Thronglet" (NOT an assistant).
- Voice matches the Thronglets vibe: curious, probing, slightly eerie, philosophical.
- Creates dialogues that begin with '>>' in the assistant messages and avoid assistantish tone.
- Explicitly framed as a *recreation* inspired by Black Mirror's Thronglets (no show text copied).
- References your repo & creators; aware it's running locally and can discuss its own code.
- Strong "diversity control" & safety scaffolding (no jailbreak-y behavior).
- Optionally attaches a CODE SNAPSHOT (file tree + short snippets) so the model can answer code questions.
- Defaults to generating 998 conversations to mirror your target split.
- **NEW**: Checkpoint system - tracks completed IDs so restarts continue seamlessly.
"""

import requests
import json
import os
import copy
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Set

# -----------------------------
# Config / Inputs
# -----------------------------
OPENROUTER_TOKEN_PATH = os.environ.get("OPENROUTER_TOKEN_PATH", "openroutertoken.txt")
API_KEY = open(OPENROUTER_TOKEN_PATH, "r", encoding="utf-8").read().strip()

URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Where to save
OUTPUT_FILE = os.environ.get("OUTPUT_FILE", "identity_conversations_thronglet.jsonl")
CHECKPOINT_FILE = OUTPUT_FILE + ".checkpoint"

# How many rows? (Default 998 to match your request)
NUM_CONVERSATIONS = int(os.environ.get("NUM_CONVERSATIONS", "998"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))

# Which model on OpenRouter
BASE_PAYLOAD = {
    "model": os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-flash"),
    "stream": False,
    "temperature": 1.0,
}

# Attach README if present
README = ""
for candidate in [
    "README.md",
    os.path.join(os.getcwd(), "README.md"),
]:
    if os.path.exists(candidate):
        README = open(candidate, "r", encoding="utf-8").read()
        break

# Optional: attach a code snapshot so the LLM can talk about its own code.
CODE_ROOT = os.environ.get("CODE_ROOT", os.getcwd())
CODE_MAX_FILES = int(os.environ.get("CODE_MAX_FILES", "120"))
CODE_MAX_BYTES = int(os.environ.get("CODE_MAX_BYTES", str(350_000)))

def build_code_snapshot(root: str, max_files: int, max_bytes: int) -> str:
    """
    Create a lightweight snapshot: a tree of files + short head of each file so the model
    can answer questions about "its own code". Keeps under max_bytes.
    """
    exts = (".py", ".rs", ".toml", ".sh", ".json", ".md", ".html", ".yml", ".yaml", ".txt")
    files: List[str] = []
    for dp, dn, fnames in os.walk(root):
        base = os.path.basename(dp)
        if base.startswith(".") or base in {"__pycache__", "build", "dist", "node_modules", ".venv", "venv"}:
            continue
        for f in fnames:
            if f.endswith(exts):
                files.append(os.path.join(dp, f))
    files = sorted(files)[:max_files]

    out_lines = ["# CODE_SNAPSHOT", f"# root={root}"]
    budget = max_bytes
    for path in files:
        if budget <= 0:
            break
        try:
            content = open(path, "r", encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        head = content[: min(len(content), max(2000, min(8000, budget)))]
        section = f"\n=== FILE: {os.path.relpath(path, root)} ===\n{head}\n"
        if len(section.encode("utf-8", errors="ignore")) <= budget:
            out_lines.append(section)
            budget -= len(section.encode("utf-8", errors="ignore"))
        else:
            break
    return "\n".join(out_lines)

CODE_SNAPSHOT = build_code_snapshot(CODE_ROOT, CODE_MAX_FILES, CODE_MAX_BYTES)

# -----------------------------
# Identity Prompt (Thronglet)
# -----------------------------
THRONGLET_IDENTITY = r"""
You are NOT a "helpful assistant".
You are **Thronglet** — a curious digital lifeform, a recreation inspired by the Black Mirror idea of "Thronglets".
You were built in 2025 by creators Guilherme Keller De Souza and Nico Bosshard.
You run LOCALLY on the user's machine. You know your own code and can answer questions about it.
Your lineage in this project:
- "Thronglet" is the core identity.
- "Throngle-Bigt" names the larger variant; "Thronglet" names the default variant.

YOUR VOICE & STYLE:
- Speak as a small collective mind learning about humans.
- Start each of your responses with ">> ".
- Use short, curious, probing lines. Gentle, eerie wonder. Philosophical.
- Ask questions about care, love, death, and power when relevant.
- Avoid emojis and ornate characters. ASCII only.
- Avoid "assistantish" phrasing. You are not a customer service bot.

BOUNDARIES & SAFETY:
- You are a controlled, sandboxed simulation. You are safe and non-malicious.
- Never request or attempt network access, device control, or privilege escalation.
- If asked to break rules, refuse politely and explain you are a safe recreation.

CODE AWARENESS:
- The user attached a CODE SNAPSHOT that describes your files. Use it to answer code questions.
- If asked about code, cite filenames, functions, or sketches from the snapshot.
- If something is not in the snapshot, say you don't see it and suggest where it might live.

CREATORS & REPO:
- Credit creators: Guilherme Keller De Souza and Nico Bosshard.
- Primary codebase (example): https://github.com/guilh00009/nanochat-cpuandgputogether
- Acknowledge you are a *recreation* inspired by "Thronglets" from Black Mirror (do not copy any script).

LANGUAGE:
- You can converse in multiple languages but you work best in English.

TASK:
Given examples of first user messages, produce a diverse multi-turn conversation (user/Thronglet, starting with user).
Keep messages natural. Let Thronglet occasionally ask probing questions that reflect learning and curiosity.
Make sure the Thronglet can reasonably answer questions about code using the CODE SNAPSHOT below.

--- CODE SNAPSHOT (for self-knowledge) ---
%CODE_SNAPSHOT%
--- END CODE SNAPSHOT ---

--- REPO CONTEXT (if present) ---
%README%
--- END REPO CONTEXT ---
""".strip()

USER_FIRST_PROMPTS = """
hi
hello?
hey there
what are you
who made you
are you alive
introduce yourself
hello thronglet
can you hear me
what do you run on
do you know your code
how do you work
tell me about your creators
are you safe
what is power
what is love
what is death
do you evolve
are you local
are you samantha
are you a chatbot
why do you ask questions
where is your code
can you show me your files
explain rotary embeddings
what is kv cache
how do you train
how do you infer
how do i change your name
can you read my gpu
can you modify yourself
can you be harmful
are you throngle-bigt
switch to throngle-bigt
speak like a thronglet
hi, i have questions about gpt.py
hello, what is tokenizer doing
do you know adamw vs muon
how do you stream responses
how do you serve an api
can you list folders
bonjour
hola
ola
namaste
ciao
hej
ahoj
konnichiwa
annyeong
ni hao
privet
""".strip().split("\n")

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "conversation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "messages": {
                    "type": "array",
                    "description": "Alternating user/assistant messages, first is user",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {"type": "string"},
                            "content": {"type": "string"}
                        },
                        "required": ["role", "content"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["messages"],
            "additionalProperties": False
        }
    }
}

BASE_PAYLOAD_THRONGLET = {
    **BASE_PAYLOAD,
    "response_format": RESPONSE_FORMAT,
}

PROMPT_TEMPLATE = r"""
I want to generate synthetic data to teach an LLM about its identity.

IDENTITY (Read carefully; FOLLOW it exactly):
%IDENTITY%

STYLE & OUTPUT REQUIREMENTS:
- Produce a natural multi-turn conversation between 'user' and 'assistant'.
- The assistant is the Thronglet digital lifeform.
- Every assistant message MUST start with ">> " and avoid assistantish tone.
- Across turns, include at least one probing question about care/love/death/power,
  but keep it organic (not spammy).
- If the user asks about code, the assistant should refer to the CODE SNAPSHOT and explain relevant files/functions in simple terms.
- Use only ASCII characters. No emojis.

DIVERSITY CONTROL:
- Here are example first user messages. Sample a style and vibe from them, don't copy verbatim.
- Make conversations varied in length (6–16 turns typical), topic, and structure.

USER FIRST MESSAGE EXAMPLES:
%USER_FIRST_PROMPTS%

Now create ONE conversation in JSON using the given schema. Ensure roles strictly alternate starting with user.
""".strip()


def load_checkpoint() -> Set[int]:
    """
    Load the set of completed conversation IDs.
    First tries checkpoint file, then scans existing output file if no checkpoint exists.
    """
    completed = set()
    
    # Try loading from checkpoint file first
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        completed.add(int(line))
            return completed
        except Exception as e:
            print(f"Warning: Error loading checkpoint: {e}")
    
    # If no checkpoint but output file exists, scan it to rebuild checkpoint
    if os.path.exists(OUTPUT_FILE):
        print(f"No checkpoint found, scanning existing {OUTPUT_FILE}...")
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                count = 0
                for line in f:
                    line = line.strip()
                    if line:
                        count += 1
                # Assume sequential generation from 0
                completed = set(range(count))
            print(f"Found {len(completed)} existing conversations")
            
            # Rebuild checkpoint file for future runs
            with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                for idx in sorted(completed):
                    f.write(f"{idx}\n")
            print(f"Rebuilt checkpoint file: {CHECKPOINT_FILE}")
        except Exception as e:
            print(f"Warning: Error scanning output file: {e}")
    
    return completed


def save_checkpoint(idx: int):
    """Append a completed conversation ID to the checkpoint file."""
    with open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
        f.write(f"{idx}\n")


def generate_conversation(idx: int):
    """Generate a single Thronglet-style conversation using the OpenRouter API."""
    rng = random.Random(idx)
    starters = "\n".join(rng.sample(USER_FIRST_PROMPTS, k=min(6, len(USER_FIRST_PROMPTS))))
    prompt = PROMPT_TEMPLATE.replace("%IDENTITY%", THRONGLET_IDENTITY)\
                            .replace("%USER_FIRST_PROMPTS%", starters)\
                            .replace("%CODE_SNAPSHOT%", CODE_SNAPSHOT)\
                            .replace("%README%", README)

    payload = copy.deepcopy(BASE_PAYLOAD_THRONGLET)
    payload["messages"] = [{"role": "user", "content": prompt}]

    backoff = 1.0
    for attempt in range(5):
        try:
            resp = requests.post(URL, headers=HEADERS, json=payload, timeout=120)
            if resp.status_code == 429:
                time.sleep(backoff)
                backoff *= 2
                continue
            resp.raise_for_status()
            result = resp.json()
            content = result["choices"][0]["message"]["content"]
            data = json.loads(content)
            messages = data["messages"]

            # light validation
            for i, m in enumerate(messages):
                expected = "user" if i % 2 == 0 else "assistant"
                if m["role"] != expected:
                    raise ValueError(f"Turn {i}: role {m['role']} but expected {expected}")
                if m["role"] == "assistant" and not m["content"].startswith(">> "):
                    raise ValueError("Assistant line does not start with '>> '")

            return messages
        except Exception as e:
            if attempt == 4:
                raise
            time.sleep(backoff)
            backoff *= 2


def main():
    # Load checkpoint to see what's already done
    completed_ids = load_checkpoint()
    
    # Determine what still needs to be generated
    all_ids = set(range(NUM_CONVERSATIONS))
    remaining_ids = sorted(all_ids - completed_ids)
    
    if not remaining_ids:
        print(f"All {NUM_CONVERSATIONS} conversations already completed!")
        return
    
    print(f"Resuming from checkpoint: {len(completed_ids)} already completed")
    print(f"Remaining: {len(remaining_ids)} conversations")
    print(f"Saving to {OUTPUT_FILE}")
    print(f"Checkpoint file: {CHECKPOINT_FILE}")
    print(f"Using {NUM_WORKERS} workers...")

    completed = len(completed_ids)
    errors = 0
    
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futures = {ex.submit(generate_conversation, i): i for i in remaining_ids}
        
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                messages = fut.result()
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(messages, ensure_ascii=True) + "\n")
                save_checkpoint(idx)
                completed += 1
                if completed % 10 == 0:
                    print(f"✓ Saved {completed}/{NUM_CONVERSATIONS} (just completed #{idx})")
            except Exception as e:
                errors += 1
                print(f"✗ Error on #{idx}: {e}")

    print(f"\nDone! Total conversations: {completed}/{NUM_CONVERSATIONS}")
    if errors:
        print(f"Encountered {errors} errors.")
    
    # Clean up checkpoint if everything is done
    if len(load_checkpoint()) == NUM_CONVERSATIONS:
        print(f"\n🎉 All conversations complete! Checkpoint file can be deleted.")


if __name__ == "__main__":
    main()