import requests
from bs4 import BeautifulSoup
import pandas as pd

def fetch_neighboring_municipalities(url):
    def get_names(soup):
        # Find the table with neighboring municipalities by looking for the header text
        table = None
        for tbl in soup.find_all("table"):
            header = tbl.find("th")
            if header and "Aangrenzende gemeenten" in header.text:
                table = tbl
                break

        # Extract municipality names
        names = []
        if table:
            links = table.find_all("a")
            for link in links:
                if 'title' in link.attrs and 'Vlag' not in link.attrs['title']:  # Exclude entries with 'Vlag'
                    name = link.attrs['title'].replace(" (gemeente)", "")
                    names.append(name)
        return names

    # Fetch the webpage content
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    names = get_names(soup)

    # If no names found, retry with modified URL
    if not names and "_(gemeente)" not in url:
        modified_url = url.replace("wiki/", "wiki/").replace("#Aangrenzende_gemeenten", "_(gemeente)#Aangrenzende_gemeenten")
        response = requests.get(modified_url)
        soup = BeautifulSoup(response.content, "html.parser")
        names = get_names(soup)
        url = modified_url
    return names, url

# Get table with municipalities from Wikipedia
table = pd.read_html("https://en.wikipedia.org/wiki/Municipalities_of_the_Netherlands")

df = table[3]
df_deleted_rows = df.iloc[::2].reset_index(drop=True)
df = df_deleted_rows.drop("Map", axis="columns")
df.columns = ["Municipality", "CBS Code", "Province", "Population", "Population Density", "Land Area"]
df = df.drop(["Population", "Population", "Population Density", "Land Area"], axis="columns")

# add to each municipality the neighboring municipalities
for idx, row in df_deleted_rows.iterrows():
    name_municipality = row["Municipality"].replace(" ", "_")

    if "(LI)" in name_municipality:
        name_municipality = "Bergen" + "_(gemeente_in_Limburg)"
        # print(name_municipality)

    if "(NH)" in name_municipality:
        name_municipality = "Bergen" + "_(Noord-Holland)"
        # print(name_municipality)

    if "Hengelo" in name_municipality:
        name_municipality = "Hengelo" + "_(Overijssel)"
        # print(name_municipality)

    if "Altena" in name_municipality:
        name_municipality = "Altena" + "_(Nederlandse_gemeente)"

    if "Bloemendaal" in name_municipality:
        name_municipality = "Bloemendaal" + "_(Noord-Holland)"

    if "Borne" in name_municipality:
        name_municipality = "Borne" + "_(Overijssel)"

    if "Heemstede" in name_municipality:
        name_municipality = "Heemstede" + "_(Noord-Holland)"

    if "Laren" in name_municipality:
        name_municipality = "Laren" + "_(Noord-Holland)"

    if "Noordwijk" in name_municipality:
        name_municipality = "Noordwijk" + "_(Zuid-Holland)"

    if "Rijswijk" in name_municipality:
        name_municipality = "Rijswijk" + "_(Zuid-Holland)"

    if "Scherpenzeel" in name_municipality:
        name_municipality = "Scherpenzeel" + "_(Gelderland)"

    if "Soest" in name_municipality:
        name_municipality = "Soest" + "_(Nederland)"

    if "The_Hague" in name_municipality:
        name_municipality = "Den_Haag"
    
    if "Wageningen" in name_municipality:
        name_municipality = "Wageningen" + "_(Nederland)"
    
    if "Zwijndrecht" in name_municipality:
        name_municipality = "Zwijndrecht" + "_(Nederland)"

    url = f"https://nl.wikipedia.org/wiki/{name_municipality}#Aangrenzende_gemeenten"
    names, modified_url = fetch_neighboring_municipalities(url)
    # print(f"Neighboring Municipalities from {modified_url}:")
    
    if names == []:
        print(f"Neighboring Municipalities from {modified_url}:")
        print(name_municipality)

    # add a new column to df with the neighboring municipalities
    df.loc[idx, "Neighboring Municipalities"] = ', '.join(names)

