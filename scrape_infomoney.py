import requests
from bs4 import BeautifulSoup
import pandas as pd
import time


def scrape_infomoney_markets():
    """
    Scrapes the InfoMoney Markets page to extract titles,
    links, and the main text from each news article.
    """
    # Main URL for the Markets section
    main_url = "https://www.infomoney.com.br/mercados/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    print(f"Accessing the main page: {main_url}")

    try:
        # 1. Get all the news links from the main page
        response = requests.get(main_url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes

        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all news links within <h2> tags
        # This is the most common structure for headlines on the page
        news_links_tags = soup.select('h2 a[href]')

        if not news_links_tags:
            print("No news links found with the selector 'h2 a[href]'. Please check the website structure.")
            return

        articles_to_scrape = []
        for a_tag in news_links_tags:
            title = a_tag.get_text(strip=True)
            link = a_tag['href']
            # Ensure it's a valid link and avoid duplicates
            if link and link.startswith('https://') and link not in [d['link'] for d in articles_to_scrape]:
                articles_to_scrape.append({'title': title, 'link': link})

        print(f"Found {len(articles_to_scrape)} news links to scrape.")

        # 2. Visit each link to extract the main text
        all_news_data = []
        for article in articles_to_scrape:
            try:
                print(f"Scraping: {article['title']}")

                article_response = requests.get(article['link'], headers=headers, timeout=10)
                article_response.raise_for_status()

                article_soup = BeautifulSoup(article_response.content, 'html.parser')

                # The main content is usually inside an <article> tag with the class 'im-article'
                content_container = article_soup.find('article', class_='im-article')

                if content_container:
                    # Find all paragraph tags within the content container
                    paragraphs = content_container.find_all('p')
                    # Join the text from all paragraphs to form the article body
                    main_text = '\n'.join([p.get_text(strip=True) for p in paragraphs])
                else:
                    main_text = "Content not found (page structure may have changed)."

                all_news_data.append({
                    'title': article['title'],
                    'link': article['link'],
                    'main_text': main_text
                })

                # Pause to avoid overloading the server
                time.sleep(1)

            except requests.exceptions.RequestException as e:
                print(f"  -> Error accessing the article '{article['title']}': {e}")
                continue  # Skip to the next article in case of an error

        return pd.DataFrame(all_news_data)

    except requests.exceptions.RequestException as e:
        print(f"Fatal error accessing the main page: {e}")
        return None


# --- Script Execution ---
if __name__ == "__main__":
    df_news = scrape_infomoney_markets()

    if df_news is not None and not df_news.empty:
        print("\n--- Scraping Complete! ---")
        print(f"Total of {len(df_news)} news articles collected.")

        # Display the first 5 collected articles
        print("\n--- Data Sample: ---")
        print(df_news.head())

        # Save the data to a CSV file in the current directory
        try:
            output_filename = "infomoney_market_news.csv"
            df_news.to_csv(output_filename, index=False, encoding='utf-8-sig')
            print(f"\nData successfully saved to file: '{output_filename}'")
        except Exception as e:
            print(f"\nError saving the CSV file: {e}")
    else:
        print("\nNo news articles were collected.")