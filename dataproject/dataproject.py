import requests
from lxml import html
import pandas as pd

class NyboligScraper:

    def get_pages(self, property_type=None):
        if property_type is None:
            url = 'https://www.nybolig.dk/til-salg'
        else:
            url = f'https://www.nybolig.dk/til-salg/{property_type}'

        response = requests.get(url)
        tree = html.fromstring(response.content)

        pages = tree.xpath('//span[@class="total-pages-text"]/text()')[0]
        print(f'Total number of pages: {pages}')

    def scrape_data_nybolig(self, num_pages, property_type=None, file_name=None):
        addresses = []
        postcodes = []
        cities = []
        prices = []
        types = []
        rooms = []
        sizes_1 = []
        sizes_2 = []

        if property_type is None:
            url = 'https://www.nybolig.dk/til-salg'
        else:
            url = f'https://www.nybolig.dk/til-salg/{property_type}'

        for page in range(1, num_pages + 1):
            page_url = f'{url}?page={page}'
            response = requests.get(page_url)

            tree = html.fromstring(response.content)
            tiles = tree.xpath('//div[@class="tile__info"]')

            for tile in tiles:
                address = tile.xpath('.//p[@class="tile__address"]/text()')
                price = tile.xpath('.//p[@class="tile__price"]/text()')
                mix = tile.xpath('.//p[@class="tile__mix"]/text()')

                # Address rensning
                if address:
                    full_address = address[0].strip()
                    addresses.append(full_address)
                    city_parts = full_address.split(', ')[-1].split(' ')
                    postcodes.append(city_parts[0] if len(city_parts) > 1 else None)
                    cities.append(' '.join(city_parts[1:]) if len(city_parts) > 1 else None)
                else:
                    addresses.append(None)
                    postcodes.append(None)
                    cities.append(None)

                # Price rensning
                if price:
                    cleaned_price = ''.join(filter(str.isdigit, price[0]))
                    prices.append(int(cleaned_price))
                else:
                    prices.append(None)

                # Mix rensning
                if mix:
                    cleaned_mix = ' '.join(mix[0].split())
                    mix_parts = cleaned_mix.split(' | ')

                    types.append(mix_parts[0] if len(mix_parts) > 0 else None)
                    rooms.append(int(mix_parts[1].split()[0]) if len(mix_parts) > 1 else None)

                    if len(mix_parts) > 2:
                        size_parts = mix_parts[2].split()[0].split('/')
                        sizes_1.append(int(size_parts[0]) if len(size_parts) > 0 else None)
                        sizes_2.append(int(size_parts[1]) if len(size_parts) > 1 else None)
                    else:
                        sizes_1.append(None)
                        sizes_2.append(None)
                else:
                    types.append(None)
                    rooms.append(None)
                    sizes_1.append(None)
                    sizes_2.append(None)

        data = {
            "address": addresses,
            "postcode": postcodes,
            "city": cities,
            "price": prices,
            "type": types,
            "rooms": rooms,
            "size_1": sizes_1,
            "size_2": sizes_2
        }
        df = pd.DataFrame(data)

        # Export the DataFrame to an Excel file
        if file_name is None:
            if property_type is None:
                file_name = f'scraped_data.csv'
            else:
                file_name = f'scraped_data_{property_type}.csv'
        else:
                file_name = f'{file_name}.csv'
        df.to_csv(file_name, index=False)
