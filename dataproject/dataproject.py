import requests
from lxml import html
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import re

class NyboligScraper:

    #Function to parse the postal code from address
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
    #get the number of pages for the given property type
    def get_pages(self, property_type=None):
        if property_type is None:
            url = 'https://www.nybolig.dk/til-salg'
        else:
            url = f'https://www.nybolig.dk/til-salg/{property_type}'

        response = requests.get(url)
        tree = html.fromstring(response.content)

        pages = tree.xpath('//span[@class="total-pages-text"]/text()')[0]
        print(f'Total number of pages: {pages}')

    # Function to parse city from address
    def parse_city_name(self,data):
        for i in range(len(data) - 1):
            if len(data) > 2 and isinstance(data[i], str) and len(data[i]) >= 4:
                for j in range(len(data[i]) - 3):
                    if data[i][j:j+4].isdigit():
                        return " ".join(data[i+1:]) 
        return None
    #scraper
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

        #Iterate through each page to scrape property data
        for page in range(1, num_pages + 1):
            page_url = f'{url}?page={page}'
            response = requests.get(page_url)

            tree = html.fromstring(response.content)
            tiles = tree.xpath('//div[@class="tile__info"]')
            #Iterate through each property and extract relevant data
            for tile in tiles:
                address = tile.xpath('.//p[@class="tile__address"]/text()')
                price = tile.xpath('.//p[@class="tile__price"]/text()')
                mix = tile.xpath('.//p[@class="tile__mix"]/text()')

                #Address parsing
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

                # Price parsing
                if price:
                    cleaned_price = ''.join(filter(str.isdigit, price[0]))
                    prices.append(int(cleaned_price))
                else:
                    prices.append(None)

                # Mixed parsing
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
        #DataFrame with the scraped data
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
        df = df[df['price'] >= 500000] #Filter out rows with price below 500000

        #Export the DataFrame to an Excel file
        if file_name is None:
            if property_type is None:
                file_name = f'scraped_data.csv'
            else:
                file_name = f'scraped_data_{property_type}.csv'
        else:
                file_name = f'{file_name}.csv'


    #City parsing again (Function to update city name if it is missing)
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
    #Calculate descriptive statistics for a given column

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
    #OLS regression
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
    #Remove outliers from the dataset based on the given column and standard error threshold
    def remove_outliers(self, column_name, threshold):
        # Calculate the z-score of the data
        z_scores = (self.data[column_name] - self.data[column_name].mean()) / self.data[column_name].std()
        # Remove outliers above the threshold
        self.data = self.data[z_scores <= threshold]
    #Function to find the minimum and maximum average property prices by postcode
    def min_max_postcode(self, data=None):
        if data is None:
            data = self.data
        mean_prices_by_postcode = data.groupby('postcode')['price'].mean()
        min_postcode = mean_prices_by_postcode.idxmin()
        min_price = mean_prices_by_postcode.loc[min_postcode]
        min_city = data[data['postcode'] == min_postcode]['city'].iloc[0]
        max_postcode = mean_prices_by_postcode.idxmax()
        max_price = mean_prices_by_postcode.loc[max_postcode]
        max_city = data[data['postcode'] == max_postcode]['city'].iloc[0]
        return (min_price, min_postcode, min_city), (max_price, max_postcode, max_city)





