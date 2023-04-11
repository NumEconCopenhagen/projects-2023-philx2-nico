def keep_regs(df, regs):
    """ Example function. Keep only the subset regs of regions in data.

    Args:
        df (pd.DataFrame): pandas dataframe 

    Returns:
        df (pd.DataFrame): pandas dataframe

    """ 
    
        # Export the DataFrame to an Excel file
        if file_name is None:
            if property_type is None:
                file_name = f'scraped_data.xlsx'
            else:
                file_name = f'scraped_data_{property_type}.xlsx'
        else:
            if property_type is None:
                file_name = f'scraped_data_{file_name}.xlsx'
            else:
                file_name = f'scraped_data_{property_type}_{file_name}.xlsx'
        df.to_excel(file_name, index=False)
    return df