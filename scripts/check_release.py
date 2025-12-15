#!/usr/bin/env python3
"""
Check the status of the latest release pipeline for the autosub repository.
Prints whether it is successful, failed, or still in progress.
Shows error logs if the release pipeline fails.

Usage:
    python check_release.py [--token GITHUB_TOKEN]

    The GitHub token can also be set via the GITHUB_TOKEN or GITHUB_API_KEY
    environment variables. A token is required to access workflow run logs.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
import zipfile
from io import BytesIO

REPO_OWNER = "windoze"
REPO_NAME = "autosub"
WORKFLOW_NAME = "Release"
API_BASE = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}"


def get_headers(token: str | None) -> dict[str, str]:
    """Build request headers with optional authentication."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def api_request(url: str, token: str | None) -> dict:
    """Make a GET request to the GitHub API."""
    req = urllib.request.Request(url, headers=get_headers(token))
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"Error: Resource not found at {url}")
        elif e.code == 401:
            print("Error: Authentication required. Please provide a valid GitHub token.")
        elif e.code == 403:
            print("Error: Access forbidden. Check your token permissions or rate limits.")
        else:
            print(f"Error: HTTP {e.code} - {e.reason}")
        sys.exit(1)


def get_workflow_id(token: str | None) -> int:
    """Get the workflow ID for the Release workflow."""
    url = f"{API_BASE}/actions/workflows"
    data = api_request(url, token)

    for workflow in data.get("workflows", []):
        if workflow["name"] == WORKFLOW_NAME:
            return workflow["id"]

    print(f"Error: Workflow '{WORKFLOW_NAME}' not found")
    sys.exit(1)


def get_latest_run(workflow_id: int, token: str | None) -> dict:
    """Get the latest workflow run."""
    url = f"{API_BASE}/actions/workflows/{workflow_id}/runs?per_page=1"
    data = api_request(url, token)

    runs = data.get("workflow_runs", [])
    if not runs:
        print("No workflow runs found for the Release workflow.")
        sys.exit(0)

    return runs[0]


def get_failed_jobs(run_id: int, token: str | None) -> list[dict]:
    """Get all failed jobs for a workflow run."""
    url = f"{API_BASE}/actions/runs/{run_id}/jobs"
    data = api_request(url, token)

    failed_jobs = []
    for job in data.get("jobs", []):
        if job["conclusion"] == "failure":
            failed_jobs.append(job)

    return failed_jobs


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Handler that captures redirect URLs instead of following them."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Don't follow the redirect, just raise an exception with the URL
        raise urllib.error.HTTPError(
            newurl, code, f"Redirect to: {newurl}", headers, fp
        )


def get_job_logs(job_id: int, token: str | None) -> str:
    """Download and extract logs for a specific job."""
    if not token:
        return "(Logs require authentication. Please provide a GitHub token.)"

    url = f"{API_BASE}/actions/jobs/{job_id}/logs"
    headers = get_headers(token)
    req = urllib.request.Request(url, headers=headers)

    # GitHub returns a 302 redirect to a pre-signed URL for logs
    # We need to capture that URL and fetch it separately (without auth)
    opener = urllib.request.build_opener(NoRedirectHandler())

    try:
        opener.open(req)
        # If we get here without redirect, something unexpected happened
        return "(Unexpected response from logs endpoint)"
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307, 308):
            # This is expected - fetch the redirect URL (no auth needed)
            redirect_url = e.geturl()
            try:
                with urllib.request.urlopen(redirect_url) as response:
                    return response.read().decode("utf-8", errors="replace")
            except urllib.error.HTTPError as e2:
                return f"(Could not fetch logs from redirect: HTTP {e2.code})"
        else:
            return f"(Could not fetch logs: HTTP {e.code})"


def extract_error_context(logs: str, context_lines: int = 50) -> str:
    """Extract relevant error context from job logs."""
    lines = logs.split("\n")
    error_indices = []

    # Find lines containing errors
    error_keywords = ["error", "Error", "ERROR", "failed", "Failed", "FAILED", "fatal", "Fatal"]
    for i, line in enumerate(lines):
        if any(keyword in line for keyword in error_keywords):
            error_indices.append(i)

    if not error_indices:
        # Return the last portion of logs if no explicit errors found
        return "\n".join(lines[-context_lines:])

    # Collect unique context around errors
    context_set = set()
    for idx in error_indices:
        start = max(0, idx - 5)
        end = min(len(lines), idx + 10)
        for i in range(start, end):
            context_set.add(i)

    # Sort and limit the output
    sorted_indices = sorted(context_set)
    if len(sorted_indices) > context_lines:
        sorted_indices = sorted_indices[-context_lines:]

    return "\n".join(lines[i] for i in sorted_indices)


def format_duration(start: str, end: str | None) -> str:
    """Calculate and format the duration between two ISO timestamps."""
    from datetime import datetime

    start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))

    if end:
        end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
    else:
        from datetime import timezone
        end_dt = datetime.now(timezone.utc)

    duration = end_dt - start_dt
    total_seconds = int(duration.total_seconds())

    if total_seconds < 60:
        return f"{total_seconds}s"
    elif total_seconds < 3600:
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes}m {seconds}s"
    else:
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        return f"{hours}h {minutes}m"


def main():
    parser = argparse.ArgumentParser(
        description="Check the status of the latest release pipeline"
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_API_KEY"),
        help="GitHub personal access token (or set GITHUB_TOKEN/GITHUB_API_KEY env var)",
    )
    args = parser.parse_args()

    print(f"Checking release pipeline for {REPO_OWNER}/{REPO_NAME}...\n")

    # Get workflow ID
    workflow_id = get_workflow_id(args.token)

    # Get latest run
    run = get_latest_run(workflow_id, args.token)

    run_id = run["id"]
    status = run["status"]
    conclusion = run.get("conclusion")
    run_number = run["run_number"]
    html_url = run["html_url"]
    head_branch = run.get("head_branch", "unknown")
    event = run.get("event", "unknown")
    created_at = run["created_at"]
    updated_at = run.get("updated_at", created_at)

    print(f"Run #{run_number}")
    print(f"Branch: {head_branch}")
    print(f"Trigger: {event}")
    print(f"URL: {html_url}")
    print()

    if status == "completed":
        duration = format_duration(created_at, updated_at)
        if conclusion == "success":
            print(f"✅ Status: SUCCESS (completed in {duration})")
        elif conclusion == "failure":
            print(f"❌ Status: FAILED (after {duration})")
            print("\n" + "=" * 60)
            print("FAILED JOBS:")
            print("=" * 60)

            failed_jobs = get_failed_jobs(run_id, args.token)

            if not failed_jobs:
                print("No failed jobs found (workflow may have been cancelled).")
            else:
                for job in failed_jobs:
                    print(f"\n📋 Job: {job['name']}")
                    print(f"   URL: {job['html_url']}")

                    # Show failed steps
                    for step in job.get("steps", []):
                        if step.get("conclusion") == "failure":
                            print(f"   ❌ Failed step: {step['name']}")

                    # Fetch and display logs
                    print("\n   --- Error Logs ---")
                    logs = get_job_logs(job["id"], args.token)
                    error_context = extract_error_context(logs)
                    # Indent the logs
                    indented_logs = "\n".join(f"   {line}" for line in error_context.split("\n"))
                    print(indented_logs)
                    print("   --- End of Logs ---\n")
        elif conclusion == "cancelled":
            print(f"⚠️  Status: CANCELLED (after {duration})")
        else:
            print(f"⚠️  Status: {conclusion.upper()} (after {duration})")
    else:
        # Still in progress
        duration = format_duration(created_at, None)
        if status == "queued":
            print(f"⏳ Status: QUEUED (waiting for {duration})")
        elif status == "in_progress":
            print(f"🔄 Status: IN PROGRESS (running for {duration})")
        else:
            print(f"⏳ Status: {status.upper()} ({duration})")

        # Show job status
        url = f"{API_BASE}/actions/runs/{run_id}/jobs"
        jobs_data = api_request(url, args.token)

        print("\nJob Status:")
        for job in jobs_data.get("jobs", []):
            job_status = job["status"]
            job_conclusion = job.get("conclusion", "")

            if job_status == "completed":
                if job_conclusion == "success":
                    emoji = "✅"
                elif job_conclusion == "failure":
                    emoji = "❌"
                else:
                    emoji = "⚠️"
            elif job_status == "in_progress":
                emoji = "🔄"
            else:
                emoji = "⏳"

            status_text = job_conclusion if job_conclusion else job_status
            print(f"  {emoji} {job['name']}: {status_text}")

    print()
    return 0 if conclusion == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
