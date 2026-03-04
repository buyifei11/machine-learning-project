import wbgapi as wb
import pandas as pd

# Define default indicators to fetch
DEFAULT_INDICATORS = {
    'NY.GDP.PCAP.CD': 'GDP per capita',
    'NY.GDP.MKTP.KD.ZG': 'GDP growth',
    'SL.UEM.TOTL.ZS': 'Unemployment',
    'FP.CPI.TOTL.ZG': 'Inflation',
    'SP.DYN.LE00.IN': 'Life expectancy'
}

def fetch_world_bank_data(indicators=None, mrv=1):
    """
    Fetches economic indicators from the World Bank API for all countries.
    
    Returns a DataFrame with columns:
      'id'        : ISO3 country code
      'economy'   : Country name
      + one column per indicator (human-readable name)
    """
    if indicators is None:
        indicators = DEFAULT_INDICATORS

    # wbgapi returns: index='economy' (ISO3), columns=['Country', *indicator_codes]
    df = wb.data.DataFrame(
        indicators.keys(),
        economy='all',
        mrv=mrv,
        labels=True
    )
    df.reset_index(inplace=True)
    # Now columns: ['economy', 'Country', 'NY.GDP.PCAP.CD', ...]

    # Build rename: indicator codes → human-readable names
    rename_mapping = {code: name for code, name in indicators.items()}
    # Also rename wbgapi's built-in columns to our standard names
    rename_mapping['economy'] = 'id'       # ISO3 code
    rename_mapping['Country'] = 'economy'  # Display name

    df.rename(columns=rename_mapping, inplace=True)

    # Filter out World Bank aggregate regions — keep only actual countries
    country_ids = set(row['id'] for row in wb.economy.list())
    df = df[df['id'].isin(country_ids)].reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = fetch_world_bank_data()
    print(df.head())
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
