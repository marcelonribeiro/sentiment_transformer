import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import xml.etree.ElementTree as ET
import argparse


def make_request(url, headers, timeout=45, max_retries=3, retry_delay=5):
    """
    A verbose and robust function to make an HTTP request with automatic retries.
    """
    for attempt in range(max_retries):
        try:
            print(f"    -> Attempting to fetch URL (attempt {attempt + 1}/{max_retries}): {url}")
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            print(f"      -> Request failed: {e}")
            if attempt + 1 < max_retries:
                print(f"      -> Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"      -> All retries failed for URL: {url}")
                return None


def get_links_from_initial_page(url, headers):
    """
    Scrapes the initial static HTML of the main markets page to get the first set of links.
    """
    print("\n--- Method 1: Reading links from the initial static page ---")
    links_with_dates = {}

    response = make_request(url, headers)
    if not response:
        print("  -> WARNING: Could not fetch the initial page. Skipping this source.")
        return links_with_dates

    try:
        soup = BeautifulSoup(response.content, 'html.parser')
        news_links_tags = soup.select('h2 a[href]')

        for a_tag in news_links_tags:
            link = a_tag['href']
            if link and "/mercados/" in link:
                links_with_dates[link] = None

        print(f"  -> Found {len(links_with_dates)} unique '/mercados/' links on the initial page.")
    except Exception as e:
        print(f"  -> ERROR: Could not parse the initial page. Error: {e}")

    return links_with_dates


def get_links_from_api():
    """
    Fetches recent article links using the 'load more' API until it stops returning new data.
    """
    print("\n--- Method 2: Fetching recent links from the live API ---")
    links_with_dates = {}
    page = 1
    api_url = "https://www.infomoney.com.br/wp-json/infomoney/v1/cards"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36',
        'Content-Type': 'application/json;charset=UTF-8'}

    while True:
        print(f"  -> Fetching page {page} from API...")
        payload = {"post_id": 2565621, "page": page, "categories": [1], "tags": []}

        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=15)
            if response.status_code != 200:
                print(f"  -> API returned status {response.status_code}. Stopping.")
                break
            json_data = response.json()
            if not json_data:
                print("  -> API returned empty list. No more pages.")
                break

            new_links_found_on_page = 0
            for item in json_data:
                link = item.get('post_permalink')
                if link and "/mercados/" in link and link not in links_with_dates:
                    links_with_dates[link] = None
                    new_links_found_on_page += 1

            if new_links_found_on_page == 0 and page > 0:
                print("  -> API page returned no new links. Stopping.")
                break
            time.sleep(1)
            page += 1
        except requests.exceptions.RequestException as e:
            print(f"  -> ERROR fetching from API: {e}. Stopping API fetch.")
            break

    print(f"  -> Found {len(links_with_dates)} unique '/mercados/' links from the API.")
    return links_with_dates


def get_links_from_local_sitemaps(sitemap_folder, cutoff_date):
    """
    Parses all .xml files in a local folder to extract article URLs and their
    last modification date, pre-filtering by the cutoff_date.
    """
    print("\n--- Method 3: Reading and pre-filtering links from local sitemap files ---")
    links_with_dates = {}
    sitemap_dir = Path(sitemap_folder)

    if not sitemap_dir.is_dir():
        print(f"  -> WARNING: Directory '{sitemap_folder}' not found. Skipping.")
        return links_with_dates

    xml_files = list(sitemap_dir.glob("*.xml"))
    if not xml_files:
        print(f"  -> WARNING: No .xml files found in '{sitemap_folder}'.")
        return links_with_dates

    print(f"  -> Found {len(xml_files)} XML files to process.")
    namespace = {'s': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

    for file_path in xml_files:
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            for url_entry in root.findall('s:url', namespace):
                loc_tag = url_entry.find('s:loc', namespace)
                lastmod_tag = url_entry.find('s:lastmod', namespace)

                if loc_tag is not None and lastmod_tag is not None:
                    url = loc_tag.text
                    if url and "/mercados/" in url:
                        lastmod_date = datetime.fromisoformat(lastmod_tag.text.replace('Z', '+00:00'))
                        if lastmod_date >= cutoff_date:
                            links_with_dates[url] = lastmod_date
        except Exception as e:
            print(f"  -> WARNING: Could not process {file_path}. Error: {e}")

    print(f"  -> Found {len(links_with_dates)} unique '/mercados/' links within the date range from local sitemaps.")
    return links_with_dates


def scrape_and_filter_articles(links_with_dates, days_ago=30):
    """
    Visits each unique URL, checks its publication date, and if it's within the
    timeframe, scrapes its title and main content, cleaning out ad blocks.
    """
    print("\n--- Final Phase: Scraping and filtering content for each unique article ---")
    all_news_data = []
    utc_now = datetime.now(timezone.utc)
    cutoff_date = utc_now - timedelta(days=days_ago)

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36'}

    total_articles = len(links_with_dates)
    for i, (link, sitemap_date) in enumerate(links_with_dates.items(), 1):
        print(f"  -> Processing URL {i}/{total_articles}: {link}")

        if sitemap_date and sitemap_date < cutoff_date:
            print("      -> Skipping (pre-filtered by sitemap date): Article is older than the date limit.")
            continue

        try:
            article_response = make_request(link, headers)
            if not article_response:
                print(f"      -> Skipping article {link} due to network errors.")
                continue

            article_soup = BeautifulSoup(article_response.content, 'html.parser')

            time_tag = article_soup.find('time', {'datetime': True})
            if not time_tag:
                print("      -> Skipping: Could not find publication date on page.")
                continue

            publication_date = datetime.fromisoformat(time_tag['datetime'].replace('Z', '+00:00'))

            if publication_date < cutoff_date:
                print("      -> Skipping: Article is older than the date limit (checked on page).")
                continue

            content_container = article_soup.find('article', class_='im-article')
            if not content_container:
                print("      -> Skipping: Article content area not found.")
                continue

            for ad_div in content_container.select('div[data-ds-component="ad"]'):
                ad_div.decompose()

            paragraphs = [
                p.get_text(separator=' ', strip=True)
                for p in content_container.find_all('p')
            ]
            clean_paragraphs = [
                p_text for p_text in paragraphs
                if p_text and "publicidade" not in p_text.lower()
            ]
            main_text = '\n'.join(clean_paragraphs)

            if not main_text:
                print("      -> Skipping: No valid main text found after cleaning.")
                continue

            print("      -> Article is valid. Saving content.")
            title = article_soup.find('title').get_text(strip=True)

            all_news_data.append({
                'publication_date': publication_date,
                'title': title,
                'link': link,
                'main_text': main_text
            })
            time.sleep(1)

        except Exception as e:
            print(f"      -> ERROR: Could not process article. {e}")

    return pd.DataFrame(all_news_data)


def main(output_path, days, sitemap_folder):
    """
    Main function to orchestrate the scraping process.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    browser_headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9,pt-BR;q=0.8,pt;q=0.7',
    }

    initial_links = get_links_from_initial_page("https://www.infomoney.com.br/mercados/", browser_headers)
    api_links = get_links_from_api()
    sitemap_links = get_links_from_local_sitemaps(sitemap_folder=sitemap_folder, cutoff_date=cutoff)

    all_unique_links_with_dates = {**initial_links, **api_links, **sitemap_links}

    if not all_unique_links_with_dates:
        print("\nNo links were found from any source. Exiting.")
    else:
        print(f"\nFound a total of {len(all_unique_links_with_dates)} unique candidate links from all sources.")
        df_news = scrape_and_filter_articles(all_unique_links_with_dates, days_ago=days)

        if df_news is not None and not df_news.empty:
            df_news = df_news.sort_values(by='publication_date', ascending=False, na_position='last').reset_index(
                drop=True)

            print("\n--- Scraping Complete! ---")
            print(f"Total of {len(df_news)} news articles collected within the date range.")
            print("\n--- Data Sample (First 5 articles): ---")
            print(df_news.head())

            try:
                df_news.to_csv(output_path, index=False, encoding='utf-8-sig')
                print(f"\nData successfully saved to file: '{output_path}'")
            except Exception as e:
                print(f"\nError saving the CSV file: {e}")
        else:
            print(f"\nNo news articles were found within the last {days} days after filtering.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape market news from the InfoMoney website.")

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the output CSV file."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of past days to scrape for articles. Default is 30."
    )
    parser.add_argument(
        "--sitemap-folder",
        type=str,
        default="data/sitemaps_infomoney",
        help="Folder where local sitemap .xml files are stored. Default is 'data/sitemaps_infomoney'."
    )

    args = parser.parse_args()

    main(output_path=args.output, days=args.days, sitemap_folder=args.sitemap_folder)