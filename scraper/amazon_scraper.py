"""
amazon_scraper.py - Amazon Review Scraper with Anti-Detection Measures.

Scrapes product reviews from Amazon with rotating user-agents, rate limiting,
proxy support, and robust error handling.
"""

import re
import time
import random
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.helpers import get_logger, retry, timer, save_csv, DATA_DIR, print_header, print_progress

logger = get_logger("scraper.amazon")


class AmazonReviewScraper:
    """Scrapes customer reviews from Amazon product pages."""

    BASE_URL = "https://www.amazon.com"
    REVIEW_URL = "https://www.amazon.com/product-reviews/{asin}?pageNumber={page}&sortBy=recent"

    def __init__(self, delay_range=(2.0, 5.0), proxies=None, max_retries=3):
        self.delay_range = delay_range
        self.proxies = proxies or []
        self.max_retries = max_retries
        self.ua = UserAgent()
        self.session = requests.Session()
        self._request_count = 0
        logger.info("Scraper initialized | delay=%.1f-%.1fs", delay_range[0], delay_range[1])

    def _get_headers(self):
        return {
            "User-Agent": self.ua.random,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        }

    def _get_proxy(self):
        if not self.proxies:
            return None
        proxy = random.choice(self.proxies)
        return {"http": proxy, "https": proxy}

    def _throttle(self):
        time.sleep(random.uniform(*self.delay_range))

    @retry(max_attempts=3, delay=2.0, backoff=2.0)
    def _fetch_page(self, url):
        self._throttle()
        self._request_count += 1
        response = self.session.get(url, headers=self._get_headers(), proxies=self._get_proxy(), timeout=15)
        response.raise_for_status()
        if "captcha" in response.text.lower():
            logger.warning("CAPTCHA detected!")
            return None
        return BeautifulSoup(response.text, "lxml")

    def _parse_review(self, element):
        try:
            review_id = element.get("id", "unknown")
            profile = element.select_one("span.a-profile-name")
            reviewer_name = profile.get_text(strip=True) if profile else "Anonymous"

            rating_el = element.select_one('i[data-hook="review-star-rating"] span')
            rating = 0.0
            if rating_el:
                match = re.search(r"(\d+\.?\d*)", rating_el.get_text())
                if match:
                    rating = float(match.group(1))

            title_el = element.select_one('a[data-hook="review-title"] span')
            title = title_el.get_text(strip=True) if title_el else ""

            date_el = element.select_one('span[data-hook="review-date"]')
            review_date = date_el.get_text(strip=True) if date_el else ""

            verified = bool(element.select_one('span[data-hook="avp-badge"]'))

            body_el = element.select_one('span[data-hook="review-body"]')
            body = body_el.get_text(strip=True) if body_el else ""

            helpful_el = element.select_one('span[data-hook="helpful-vote-statement"]')
            helpful_votes = 0
            if helpful_el:
                match = re.search(r"(\d+)", helpful_el.get_text())
                if match:
                    helpful_votes = int(match.group(1))

            return {
                "review_id": review_id, "reviewer_name": reviewer_name,
                "rating": rating, "title": title, "date": review_date,
                "verified_purchase": verified, "body": body,
                "helpful_votes": helpful_votes, "scraped_at": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error("Parse error: %s", e)
            return None

    @timer
    def scrape_product(self, asin, max_pages=10):
        print_header(f"Scraping ASIN: {asin}")
        all_reviews = []
        for page_num in range(1, max_pages + 1):
            url = self.REVIEW_URL.format(asin=asin, page=page_num)
            soup = self._fetch_page(url)
            if soup is None:
                continue
            elements = soup.select('div[data-hook="review"]')
            if not elements:
                break
            for elem in elements:
                review = self._parse_review(elem)
                if review:
                    review["asin"] = asin
                    all_reviews.append(review)
            print_progress(page_num, max_pages, prefix="Pages scraped")
        logger.info("Done | ASIN=%s | reviews=%d", asin, len(all_reviews))
        return all_reviews

    def save_reviews(self, reviews, filename="scraped_reviews.csv"):
        if not reviews:
            return pd.DataFrame()
        df = pd.DataFrame(reviews)
        save_csv(df, filename)
        return df


if __name__ == "__main__":
    scraper = AmazonReviewScraper(delay_range=(3, 7))
    reviews = scraper.scrape_product("B08N5WRWNW", max_pages=5)
    scraper.save_reviews(reviews)
