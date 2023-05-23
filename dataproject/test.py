import pandas as pd
import re
from collections import defaultdict
from pathlib import Path
import csv
from lxml import html
import requests
import requests
from lxml import html
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import re

class HouseListingsScraper(object):
    def __init__(self,base_url:str,property_type:str,page_num:int):
        self.base_url = base_url
        self.current_page = self.base_url[12:].split(".")[0]
        self.property_type = "/" + property_type if property_type else ""
        self.page_num = f"?page={page_num}" if page_num else ""
        self.url = self.base_url + self.property_type + self.page_num
        print(self.url)
        self._response = None
        self.session = requests.Session()

    def response(self):
        if self._response is None:
            r = self.session.get(self.url)
            r.raise_for_status()
            self._response = html.fromstring(r.text)
        return self._response
    
    def get_element(self,xpath):
        return self.response().xpath(xpath)

    def parse_listings(self):
        return self.get_element("//div[@class='tile__info']")

    def get_pages(self, property_type=None):
        if property_type is None:
            url = 'https://www.nybolig.dk/til-salg'
        else:
            url = f'https://www.nybolig.dk/til-salg/{property_type}'

        response = requests.get(url)
        tree = html.fromstring(response.content)

        pages = tree.xpath('//span[@class="total-pages-text"]/text()')[0]
        print(f'Total number of pages: {pages}')


class NyBoligParser(object):
    def __init__(self,scraped_data):
        self.scraped_data = scraped_data
        # print(self.scraped_data)
        self.unparsed = list()
        self.listing_data_dict = defaultdict(list)
        
    def parse_listings(self):
        for index,element in enumerate(self.scraped_data):
            ele = element.getchildren()
            items = [ele[0].text.strip(),ele[1].text.strip(),ele[2].text.strip()]
            self.parse_listing_elements(items)
            self.unparsed.append(items)

        self.raw_data


    def raw_data(self):
        # Sanity Check
        #print(self.unparsed)

        # Basic file name
        filename = 'raw_output.csv'
        # Open the file in write mode and specify newline='' to prevent extra blank lines
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(self.unparsed)

    def parse_listing_elements(self, listing):
        item, price, details = listing
        self.append_location_data(item)
        cleaned_price = ''.join(filter(str.isdigit, price))
        price = int(cleaned_price)
        self.listing_data_dict['price'].append(price)
        self.append_structural_data(details)


    def parse_city_name(self,data):
        #Some cities have multiple words in the name (e.g. lolland falster). Some listings also do not have postal code.
        # It loops through each element (expect the last element (e.g. "falster" in "lolland falster" is not relevant))
        # ensuring that we remove the postal code of the city name and saves everything after the postal code as the city name
        # We also ensure that we dont get floor name 
        for i in range(len(data) - 1):
            if len(data) > 2 and isinstance(data[i], str) and len(data[i]) >= 4:
                for j in range(len(data[i]) - 3):
                    if data[i][j:j+4].isdigit():
                        return " ".join(data[i+1:])

        return None

    def parse_postal_code(self, address):
        #use strip to remove white space
        full_address = address.strip()
        #use regex to find postal code pattern (4 numbers in a row that follows Danish postal code conventions)
        postal_code_pattern = r'\b\d{4}\b'
        postal_code_match = re.search(postal_code_pattern, full_address)
        #if postal_code_match is true we define the postal code. (some listings do not contain postal code)
        if postal_code_match:
            postal_code = postal_code_match.group()
        else:
            postal_code = None

        return postal_code

    def append_location_data(self,full_address):
        #we define a temporary dictionary to hold data
        data = {
            "full_address": full_address,
            "postal_code": self.parse_postal_code(full_address),
            #We split location data by ',' to get postal code and city. We then split it again by spaces to get city. 
            "city": self.parse_city_name(full_address.split(",")[-1].split())
        }
        #the line below uses the update function to add information to the dictionary as we iterate over the listings. 
        self.listing_data_dict.update({key: self.listing_data_dict.get(key, []) + [data[key]] for key in data})

    def append_structural_data(self, item):
        data = item.split()
        sq_m, sq_m2 = None, None
        #relavant data always has length larger than 5
        # we check for each listing if there is a basement by checking for "/"
        #if there is a basement we populate sq_m2 representing the value for basement square meter
        #lastly, we convert the output to integer
        if len(data) > 5:
            try:
                if "/" in data[5]:
                    sq_m = int(data[5].split("/")[0])
                    sq_m2 = int(data[5].split("/")[1])
                else:
                    sq_m = int(data[5])
                    sq_m2 = None
            except ValueError:

                sq_m, sq_m2 = None, None
        #some listings do not have a room number
        #like before we use a try and except method for setting the rooms variable. 
        rooms = None
        try:
            rooms = int(data[2])
        except ValueError:
            rooms = None


        self.listing_data_dict.update({k: self.listing_data_dict.get(k, []) + [v] for k, v in {
            "type": data[0], 
            "rooms": rooms, 
            "sq_m": sq_m, 
            "sq_m_cellar": sq_m2
        }.items()})


    def data_to_df(self):
        self.df = pd.DataFrame.from_dict(self.listing_data_dict)
        self.df[self.df['price'] >= 500000]
        self.df.loc[self.df['city'].isnull(), 'city'] = self.df[self.df['city'].isnull()].apply(self.update_city, axis=1)
        return self.df

    def update_city(self,row):
        address = row['full_address']
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

    def save_dataframe_to_csv(self, df):
        """Save a Pandas DataFrame to a CSV file.
    
        Args:
            df (pandas.DataFrame): A Pandas DataFrame.
    
        Returns:
            None
        """
        file_path = "output.csv"
        df.to_csv(file_path, index=None)
    
    def main(self,ind):
        self.parse_listings()
        self.save_dataframe_to_file(ind,self.data_to_df())

def scrape_page(property_type,page_num):
    # Create a scraper for the given page
    scraper = HouseListingsScraper("https://www.nybolig.dk/til-salg", property_type=property_type, page_num=page_num)
    # Scrape listings data from the page
    return scraper.parse_listings()

def main():

    lst = []

    for i in range(1,10):
        parser = NyBoligParser(scrape_page("sommerhus",i))
        parser.parse_listings()
        lst.append(parser.data_to_df())

    df = pd.concat(lst)

    df.to_csv('output2.csv', index=False)

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
    
    # Remove outliers from the dataset based on the given column and minimum number of observations per city
    def remove_outliers(self, column_name, min_city_observations, threshold):
        # Get the counts of observations per city
        city_counts = self.data.groupby('city').size().reset_index(name='counts')
        # Remove cities with less than min_city_observations observations
        cities_to_remove = city_counts[city_counts['counts'] < min_city_observations]['city']
        self.data = self.data[~self.data['city'].isin(cities_to_remove)]
        # Calculate the z-score of the data
        z_scores = (self.data[column_name] - self.data[column_name].mean()) / self.data[column_name].std()
        # Remove outliers above the threshold
        self.data = self.data[z_scores <= threshold]

    #Function to find the minimum and maximum average property prices by postcode
    def min_max_postcode(self, data=None):
        if data is None:
            data = self.data
        mean_prices_by_postcode = data.groupby('postal_code')['price'].mean()
        min_postcode = mean_prices_by_postcode.idxmin()
        min_price = mean_prices_by_postcode.loc[min_postcode]
        min_city = data[data['postal_code'] == min_postcode]['city'].iloc[0]
        max_postcode = mean_prices_by_postcode.idxmax()
        max_price = mean_prices_by_postcode.loc[max_postcode]
        max_city = data[data['postal_code'] == max_postcode]['city'].iloc[0]
        return (min_price, min_postcode, min_city), (max_price, max_postcode, max_city)