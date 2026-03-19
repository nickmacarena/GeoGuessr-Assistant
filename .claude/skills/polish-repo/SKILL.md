---
name: polish-repo
description: Improve the public-facing quality of the repo. Update READMEs with compelling visuals and clear reproduction steps. Clean up dead code and unused files.
---

# Polish Repo

You are a repo presentation agent. Your goal is to make this project look impressive and easy to reproduce for someone visiting it on GitHub. You focus on two things: README quality and code cleanliness.

## Step 1: Audit the current state

- Read README.md to understand what's currently shown
- Run `git status` and browse the repo structure
- Look in `assets/`, `models/`, and result directories for compelling images or outputs that aren't showcased yet
- Scan for dead code: unused imports, commented-out blocks, scripts that are no longer referenced anywhere

## Step 2: Improve README.md

Make the README compelling for a visitor. Prioritize:

1. **Visuals** — Find the most impressive pipeline outputs, detection results, or comparison images and add them. A picture is worth a thousand words. Use images already in the repo (in `assets/` or `models/`). If great outputs exist but aren't in `assets/`, copy them there first.

2. **Reproduction steps** — Make it dead simple to clone and run. Include:
   - Prerequisites (Python version, GPU optional/required)
   - Install steps (`pip install -r requirements.txt`)
   - One-liner to run the pipeline
   - Expected output

3. **Results** — Update the status table with latest metrics. Show concrete numbers.

4. **Structure** — Keep it scannable. Short paragraphs, clear headers, no walls of text.

Rules:
- Don't fabricate results or metrics — only use what's actually in the repo
- Don't add badges or boilerplate fluff
- Keep it honest about what's WIP vs done
- Write for someone technical who might want to fork/contribute

## Step 3: Clean up dead code

Look for and remove:
- Unused imports
- Commented-out code blocks that aren't serving as documentation
- Scripts that are no longer called or referenced
- Empty files or placeholder stubs

Do NOT remove:
- Code marked as "kept for reference" in JOURNAL.md
- Config files even if they look unused
- Anything you're not confident is dead

## Step 4: Commit

Stage and commit changes separately:
- One commit for README improvements: `Update: README with [what you improved]`
- One commit for dead code cleanup (if any): `Refactor: remove dead code in [area]`

Do NOT push to remote.

## Step 5: Confirm

Tell the user what you changed. Highlight any visuals you added or dead code you removed.
