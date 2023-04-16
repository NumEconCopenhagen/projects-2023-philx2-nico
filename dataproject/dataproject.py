import requests
from lxml import html
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import re

class NyboligScraper:

    #postcode cleaning
    def parse_postal_code(self, address):
        full_address = address.strip()
        postal_code_pattern = r'\b\d{4}\b'
        postal_code_match = re.search(postal_code_pattern, full_address)

        if postal_code_match:
            postal_code = postal_code_match.group()
            full_address = full_address.replace(postal_code, '').strip()
        else:
            postal_code = None

        return postal_code

    def get_pages(self, property_type=None):
        if property_type is None:
            url = 'https://www.nybolig.dk/til-salg'
        else:
            url = f'https://www.nybolig.dk/til-salg/{property_type}'

        response = requests.get(url)
        tree = html.fromstring(response.content)

        pages = tree.xpath('//span[@class="total-pages-text"]/text()')[0]
        print(f'Total number of pages: {pages}')

    #address cleaning
    def parse_city_name(self,data):
        for i in range(len(data) - 1):
            if len(data) > 2 and isinstance(data[i], str) and len(data[i]) >= 4:
                for j in range(len(data[i]) - 3):
                    if data[i][j:j+4].isdigit():
                        return " ".join(data[i+1:]) 
        return None

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

                # Address cleaning
                if address:
                    full_address = address[0].strip()
                    addresses.append(full_address)
                    city_parts = full_address.split(', ')[-1].split(' ')
                    postcodes.append(self.parse_postal_code(full_address))
                    #cities.append(' '.join(city_parts[1:]) if len(city_parts) > 1 else None)
                    cities.append(self.parse_city_name(full_address.split(",")[-1].split()))
                else:
                    addresses.append(None)
                    postcodes.append(None)
                    cities.append(None)

                # Price cleaning
                if price:
                    cleaned_price = ''.join(filter(str.isdigit, price[0]))
                    prices.append(int(cleaned_price))
                else:
                    prices.append(None)

                # Mix cleaning
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
        df = df[df['price'] >= 500000]  # Filter out rows with price below 500000

        # Export the DataFrame to an Excel file
        if file_name is None:
            if property_type is None:
                file_name = f'scraped_data.csv'
            else:
                file_name = f'scraped_data_{property_type}.csv'
        else:
                file_name = f'{file_name}.csv'


    #Address cleaning again
        def update_city(row):
            address = row['address']
            address_components = address.split()
            if not any(char.isdigit() for char in address_components[0]) and not any(char.isdigit() for char in address_components[-1]):
                for component in address_components[1:-1]:
                    if any(char.isdigit() for char in component) and address_components[-1].istitle():
                        for index, item in enumerate(reversed(address_components)):
                            if any(char.isdigit() for char in item):
                                start_index = len(address_components) - index
                                end_index = len(address_components)
                                my_string = ' '.join(address_components[start_index:end_index])
                                return my_string
            return row['city']

        df.loc[df['city'].isnull(), 'city'] = df[df['city'].isnull()].apply(update_city, axis=1)
        df.to_csv(file_name, index=False)
    

class NyboligAnalysis:
    def __init__(self, file_name):
        self.data = pd.read_csv(file_name)
        #self.data = file_name
    def descriptive_statistics(self, column_name):
        column = self.data[column_name]
        stats = {
            'count': column.count(),
            'mean': column.mean(),
            'std': column.std(),
            'min': column.min(),
            '25%': column.quantile(0.25),
            '50%': column.quantile(0.5),
            '75%': column.quantile(0.75),
            'max': column.max()
        }
        return pd.DataFrame(stats, index=[column_name])
    
    def OLS(self, X, y):
        # Add a constant to the independent variables
        X = sm.add_constant(X)

        # Create the OLS model
        model = sm.OLS(y, X)

        # Fit the model
        results = model.fit()

        # Print the summary of the results
        return results.summary()

    def plot_regression(self, X, y):
        # Fit the OLS model
        model = sm.OLS(y, sm.add_constant(X))
        results = model.fit()

        # Plot the data and the regression line
        fig, ax = plt.subplots()
        sns.scatterplot(x=X.iloc[:,0], y=y, ax=ax)
        sns.lineplot(x=X.iloc[:,0], y=results.predict(), color='r', ax=ax)
        ax.set_xlabel(X.columns[0])
        ax.set_ylabel(y.name)
        ax.set_title(f'Regression of {y.name} on {X.columns[0]}')
        plt.show()

    def remove_outliers(self, column_name, threshold):
        # Calculate the z-score of the data
        z_scores = (self.data[column_name] - self.data[column_name].mean()) / self.data[column_name].std()
        # Remove outliers above the threshold
        self.data = self.data[z_scores <= threshold]

    def group_mean(self, var1, var2):
        grouped = self.data.groupby(var2)[var1].mean()
        return pd.DataFrame(grouped)
    
    def min_max_prices(self):
        # Group the data by postcode and calculate the mean price
        mean_prices_by_postcode = self.group_mean('price', 'postcode')

        # Get the minimum and maximum price and their corresponding postcodes
        min_price = mean_prices_by_postcode['price'].min()
        max_price = mean_prices_by_postcode['price'].max()
        min_postcode = mean_prices_by_postcode.loc[mean_prices_by_postcode['price'].idxmin(), 'postcode']
        max_postcode = mean_prices_by_postcode.loc[mean_prices_by_postcode['price'].idxmax(), 'postcode']

        return min_price, max_price, min_postcode, max_postcode
    



