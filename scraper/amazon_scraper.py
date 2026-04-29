"""Amazon review scraper module.

Best-effort scraper for demo use.
Amazon pages can change frequently and may block automated requests,
so this module is intentionally defensive and returns an empty dataframe
if scraping fails.
"""

from __future__ import annotations

import re
from typing import List

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


def _build_headers() -> dict:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    }


def _extract_review_blocks(soup: BeautifulSoup) -> List[BeautifulSoup]:
    blocks = soup.select('[data-hook="review"]')
    if blocks:
        return blocks
    blocks = soup.select("div.review")
    return blocks


def _parse_rating(text: str) -> float:
    if not text:
        return np.nan
    match = re.search(r"([0-9.]+)\s+out of\s+5", text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return np.nan
    return np.nan


def extract_reviews_from_html(html: str) -> pd.DataFrame:
    """Parse review fields from Amazon HTML."""
    soup = BeautifulSoup(html, "html.parser")
    blocks = _extract_review_blocks(soup)

    rows = []
    for block in blocks:
        author = block.select_one('[data-hook="review-author"]')
        title = block.select_one('[data-hook="review-title"]')
        body = block.select_one('[data-hook="review-body"]')
        rating = block.select_one('[data-hook="review-star-rating"]') or block.select_one('[data-hook="cmps-review-star-rating"]')
        date = block.select_one('[data-hook="review-date"]')

        review_title = title.get_text(" ", strip=True) if title else ""
        review_body = body.get_text(" ", strip=True) if body else ""
        review_text = f"{review_title} {review_body}".strip()

        rows.append(
            {
                "reviewerID": author.get_text(" ", strip=True) if author else "unknown",
                "productID": "amazon-product",
                "reviewText": review_text,
                "rating": _parse_rating(rating.get_text(" ", strip=True)) if rating else np.nan,
                "reviewTime": date.get_text(" ", strip=True) if date else "",
                "source": "live_scrape",
            }
        )

    return pd.DataFrame(rows)


def scrape_amazon_reviews(url: str, max_attempts: int = 2, timeout: int = 15) -> pd.DataFrame:
    """Try to fetch reviews from an Amazon product page."""
    url = (url or "").strip()
    if not url:
        return pd.DataFrame(columns=["reviewerID", "productID", "reviewText", "rating", "reviewTime", "source"])

    headers = _build_headers()

    for _ in range(max_attempts):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            if response.status_code != 200:
                continue

            df = extract_reviews_from_html(response.text)
            if not df.empty:
                return df.drop_duplicates(subset=["reviewerID", "reviewText"]).reset_index(drop=True)
        except Exception:
            continue

    return pd.DataFrame(columns=["reviewerID", "productID", "reviewText", "rating", "reviewTime", "source"])


def save_scraped_reviews(df: pd.DataFrame, output_path: str = "data/scraped_reviews.csv") -> None:
    """Save scraped reviews to CSV for later analysis."""
    if df is None or df.empty:
        return
    df.to_csv(output_path, index=False)
