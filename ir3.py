"""
Assignment Question:
--------------------
For this assignment, write several small python programs to scrape simple HTML data from
several websites. Use Python with the following libraries:
• Beautiful Soup 4 (makes it easier to pull data out of HTML and XML documents)
• Requests (for handling HTTP requests from python)
• lxml (XML and HTML parser) [Note: here we use 'html.parser' instead of 'lxml' to avoid installation issues]

Example: If you are collecting data from Wikipedia “List of...” pages. 
In the following example, Wikipedia List of Nobel Laureates is used.

Task:
-----
- Scrape the winners of the Nobel Prize, 
- The year they won, 
- The subject (category), and 
- The URL of their individual Wiki page.
- Combine this into a single data table (like the provided example table).
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd

# Target Wikipedia page
URL = "https://en.wikipedia.org/wiki/List_of_Nobel_laureates"

# Add headers so Wikipedia doesn't block request
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(URL, headers=headers)

if response.status_code != 200:
    raise Exception(f"Failed to load page! Status code: {response.status_code}")

# Parse the page with html.parser
soup = BeautifulSoup(response.content, "html.parser")

# Wikipedia structure: tables with class "wikitable"
tables = soup.find_all("table", {"class": "wikitable"})

data = []

for table in tables:
    rows = table.find_all("tr")
    for row in rows[1:]:  # skip header
        cols = row.find_all("td")
        if len(cols) >= 3:
            year = cols[0].get_text(strip=True)
            subject = cols[1].get_text(strip=True)

            # Multiple winners per year/subject
            winners_links = cols[2].find_all("a")
            for link in winners_links:
                winner_name = link.get_text(strip=True)
                winner_url = "https://en.wikipedia.org" + link.get("href", "")
                data.append([winner_name, subject, year, winner_url])

# Create dataframe
df = pd.DataFrame(data, columns=["winner_name", "subject", "year", "url"])

# Display first few rows
print(df.head(10))

# Save to CSV
df.to_csv("nobel_laureates.csv", index=False)
print("Data scraped and saved to nobel_laureates.csv")
