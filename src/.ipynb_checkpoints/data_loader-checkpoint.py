"""
Data loading utilities for Spotify Song Popularity project.

This module provides functions to load and preprocess Spotify song data,
avoiding code duplication across notebooks.

Author: [Your Name]
Date: 2024
"""

import pandas as pd
import numpy as np


def load_raw_data(url=None):
    """
    Load raw Spotify songs dataset from TidyTuesday.
    
    Parameters
    ----------
    url : str, optional
        URL to the raw data CSV. If None, uses default TidyTuesday URL.
    
    Returns
    -------
    pd.DataFrame
        Raw Spotify songs dataframe with 32,833 rows and 23 columns.
    
    Example
    -------
    >>> songs_df = load_raw_data()
    >>> print(songs_df.shape)
    (32833, 23)
    """
    if url is None:
        url = 'https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv'
    
    songs_df = pd.read_csv(url)
    return songs_df


def load_cleaned_data(filepath='../data/clean_spotify_data.csv'):
    """
    Load cleaned Spotify data (output from notebook 01).
    
    This function loads the deduplicated dataset with all identifier columns
    removed and categorical types properly set.
    
    Parameters
    ----------
    filepath : str, default '../data/clean_spotify_data.csv'
        Path to the cleaned CSV file relative to notebooks/ directory.
    
    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with categorical types restored.
        Shape: (25,190 rows × 17 columns)
    
    Example
    -------
    >>> clean_df = load_cleaned_data()
    >>> print(clean_df.dtypes)
    """
    df = pd.read_csv(filepath)
    
    # Restore categorical dtypes (lost in CSV export)
    categorical_cols = ['playlist_genre', 'key', 'mode', 'release_year', 
                        'release_month', 'decades']
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    return df


def load_log_transformed_data(filepath='../data/log_transformed_spotify_data.csv'):
    """
    Load log-transformed Spotify data (output from notebook 01).
    
    This dataset includes log-transformed versions of skewed features
    (speechiness, acousticness, instrumentalness, liveness) and the
    logit-transformed target variable (pop_log).
    
    Parameters
    ----------
    filepath : str, default '../data/log_transformed_spotify_data.csv'
        Path to the log-transformed CSV file relative to notebooks/ directory.
    
    Returns
    -------
    pd.DataFrame
        Log-transformed dataframe with categorical types restored.
        Shape: (25,190 rows × 17 columns)
        
    Features
    --------
    - Categorical: playlist_genre, key, mode, release_year, release_month, decades
    - Continuous: danceability, energy, loudness, valence, tempo, duration_ms
    - Log-transformed: speechiness_log, acousticness_log, instrumentalness_log, liveness_log
    - Target: pop_log (logit-transformed popularity)
    
    Example
    -------
    >>> log_df = load_log_transformed_data()
    >>> print(log_df.columns.tolist())
    ['playlist_genre', 'key', ..., 'pop_log']
    """
    df = pd.read_csv(filepath)
    
    # Restore categorical dtypes
    categorical_cols = ['playlist_genre', 'key', 'mode', 'release_year', 
                        'release_month', 'decades']
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    return df


def prepare_modeling_data(log_df):
    """
    Prepare data for linear modeling by creating generic column names.
    
    This function creates a version of the dataset with x01, x02, ... naming
    convention commonly used in statistical modeling, along with a mapping
    dictionary to track original names.
    
    Parameters
    ----------
    log_df : pd.DataFrame
        Log-transformed dataframe with all features (from load_log_transformed_data).
    
    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        - lm_df: Dataframe with generic column names (x01, x02, ..., y)
        - col_map: Mapping between original and generic names
    
    Notes
    -----
    Column naming convention:
    - x01-x06: Categorical variables (genre, key, mode, year, month, decade)
    - x07-x16: Continuous variables (audio features)
    - y: Target variable (pop_log)
    
    Example
    -------
    >>> log_df = load_log_transformed_data()
    >>> lm_df, col_map = prepare_modeling_data(log_df)
    >>> print(col_map.head())
       original_name model_name
    0  playlist_genre       x01
    1             key       x02
    """
    # Select categorical and numeric features
    cat_vars = log_df.select_dtypes('category').columns.tolist()
    num_vars = log_df.select_dtypes('number').columns.tolist()
    
    # Remove pop_log from numeric (it's our target)
    if 'pop_log' in num_vars:
        num_vars.remove('pop_log')
    
    # Combine in order: categoricals first, then numerics, then target
    lm_df = pd.concat([log_df[cat_vars], log_df[num_vars], log_df[['pop_log']]], axis=1)
    
    # Create generic names
    old_cols = lm_df.columns.tolist()
    new_cols = [f'x{i:02d}' for i in range(1, len(old_cols))] + ['y']
    
    # Create mapping dataframe
    col_map = pd.DataFrame({
        'original_name': old_cols,
        'model_name': new_cols
    })
    
    # Rename columns
    lm_df.columns = new_cols
    
    return lm_df, col_map


def get_feature_names(col_map):
    """
    Get dictionary mapping model names (x01, x02) to original feature names.
    
    Parameters
    ----------
    col_map : pd.DataFrame
        Mapping dataframe from prepare_modeling_data()
    
    Returns
    -------
    dict
        Dictionary with model_name as keys, original_name as values.
        
    Example
    -------
    >>> lm_df, col_map = prepare_modeling_data(log_df)
    >>> name_dict = get_feature_names(col_map)
    >>> print(name_dict['x08'])
    'energy'
    """
    return dict(zip(col_map['model_name'], col_map['original_name']))


def map_coefficient_name(coef_name, col_map_dict):
    """
    Map coefficient names back to original feature names.
    
    Handles both regular features and categorical dummy variables.
    
    Parameters
    ----------
    coef_name : str
        Coefficient name from statsmodels output (e.g., 'x08' or 'x01[T.rock]')
    col_map_dict : dict
        Mapping dictionary from get_feature_names()
    
    Returns
    -------
    str
        Original feature name (e.g., 'energy' or 'playlist_genre[T.rock]')
        
    Example
    -------
    >>> name_dict = get_feature_names(col_map)
    >>> map_coefficient_name('x08', name_dict)
    'energy'
    >>> map_coefficient_name('x01[T.rock]', name_dict)
    'playlist_genre[T.rock]'
    """
    # Direct match for continuous variables
    if coef_name in col_map_dict:
        return col_map_dict[coef_name]
    
    # Categorical dummy variables (e.g., x01[T.rock] -> playlist_genre[T.rock])
    if '[' in coef_name:
        base_var = coef_name.split('[')[0]
        category = coef_name.split('[')[1]
        if base_var in col_map_dict:
            return f"{col_map_dict[base_var]}[{category}"
    
    # Return as-is if no mapping found (e.g., 'Intercept')
    return coef_name
