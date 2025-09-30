import streamlit as st
import pandas as pd

@st.cache_data(ttl=600)
def cache_fetch(_func, *args, **kwargs):
    """Wrapper to cache API calls."""
    return _func(*args, **kwargs)

def to_dataframe(results, cols):
    """Convert list of dicts to dataframe with selected cols."""
    df = pd.DataFrame(results)
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    return df
