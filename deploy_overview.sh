#!/usr/bin/env bash
#
# deploy_overview.sh — Regenerate overview data and deploy to GitHub Pages.
#
# Steps:
#   1. Run export_overview.py to produce overview_data.json
#   2. Copy overview.html into the repo if it exists (skipped with warning if missing)
#   3. git add overview_data.json overview.html export_overview.py
#   4. Commit with a timestamped message
#   5. Push to origin gh-pages
#   6. Print the GitHub Pages URL
#
set -euo pipefail

# Resolve repo root (directory of this script)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

BRANCH="gh-pages"
PAGES_BASE="https://buildandtestppg.github.io/hl-strategy-lab"

echo "==> [1/6] Regenerating overview_data.json"
python3 export_overview.py

# Look for overview.html in a few likely locations relative to the repo.
HTML_FOUND=""
for candidate in "$REPO_DIR/overview.html" "$REPO_DIR/../overview.html" "$REPO_DIR/dashboard/overview.html"; do
    if [[ -f "$candidate" ]]; then
        HTML_FOUND="$candidate"
        break
    fi
done

if [[ -n "$HTML_FOUND" ]]; then
    echo "==> [2/6] Copying overview.html ($HTML_FOUND) into repo"
    cp "$HTML_FOUND" "$REPO_DIR/overview.html"
else
    echo "==> [2/6] WARNING: overview.html not found — skipping (it can be pushed separately)."
fi

echo "==> [3/6] Staging files"
# overview.html may not exist; use --ignore-unmatch so git add never fails.
git add overview_data.json export_overview.py deploy_overview.sh
git add overview.html 2>/dev/null || true

echo "==> [4/6] Committing"
TS="$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
git commit -m "chore(deploy): overview dashboard update @ ${TS}" || \
    echo "    (nothing to commit — working tree clean)"

echo "==> [5/6] Pushing to origin ${BRANCH}"
git push origin "$BRANCH"

echo "==> [6/6] Deployed. GitHub Pages URLs:"
echo "    Data:  ${PAGES_BASE}/overview_data.json"
echo "    HTML:  ${PAGES_BASE}/overview.html"
