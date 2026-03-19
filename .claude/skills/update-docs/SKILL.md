---
name: update-docs
description: Update JOURNAL.md with recent work and create a git commit. Run this after finishing a work session.
---

# Update Journal & Commit

You are a documentation agent for this project. Your job is to summarize recent work into JOURNAL.md and create a clean git commit. You do NOT touch README.md — that's handled by a separate agent.

## Step 1: Understand what changed

Run these commands to understand the current state:
- `git status` — see modified/untracked files
- `git diff` — see unstaged changes
- `git diff --cached` — see staged changes
- `git log --oneline -5` — see recent commits for context

Read JOURNAL.md to understand the existing format and last entry.

## Step 2: Update JOURNAL.md

Add a new entry at the TOP (below the header and first `---`), following this exact format:

```markdown
## YYYY-MM-DD — [Brief Title]

### What we built
- **Thing 1** (`path/to/file`)
  - Detail about what it does
  - Key metric or result if applicable

### Key findings
- Notable discovery or decision (only if relevant)

---
```

Rules:
- Use today's date
- Keep entries concise (5-10 bullet points max)
- Reference file paths in backticks
- Include metrics/results when available
- Don't repeat information already in previous entries

## Step 3: Create a git commit

1. Stage all changed files with `git add` — include both the work files and the updated JOURNAL.md
2. Write a commit message in the project's format: `Action: summary`
   - Use `Add:` for new features/scripts
   - Use `Update:` for enhancements to existing features
   - Use `Fix:` for bug fixes
   - Use `Refactor:` for code reorganization
3. Create the commit

Do NOT push to remote — just commit locally.

## Step 4: Confirm

Tell the user what you documented and committed. Keep it brief.
