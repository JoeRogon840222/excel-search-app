import streamlit as st
import pandas as pd
import numpy as np
import io
from typing import Optional, List

# Set page configuration
try:
    st.set_page_config(
        page_title="Excel Data Search",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception as e:
    pass  # In case page config was already set

def load_excel_file(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Load Excel file and return DataFrame
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        pandas DataFrame or None if error
    """
    try:
        # Read Excel file
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        elif uploaded_file.name.endswith('.xls'):
            df = pd.read_excel(uploaded_file, engine='xlrd')
        else:
            st.error("Unsupported file format. Please upload .xlsx or .xls files.")
            return None
        
        # Clean up data types to prevent Arrow conversion issues
        for col in df.columns:
            # Convert mixed type columns to string to avoid Arrow errors
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str)
            
        return df
    except Exception as e:
        st.error(f"Error reading Excel file: {str(e)}")
        return None

def get_searchable_columns(df: pd.DataFrame) -> List[str]:
    """
    Get list of columns that contain text data suitable for searching
    
    Args:
        df: pandas DataFrame
        
    Returns:
        List of column names
    """
    searchable_cols = []
    for col in df.columns:
        # Check if column contains text data
        if df[col].dtype == 'object' or df[col].dtype.name == 'string':
            searchable_cols.append(col)
        # Also include numeric columns that might contain searchable data
        elif df[col].dtype in ['int64', 'float64']:
            searchable_cols.append(col)
    
    return searchable_cols

def search_dataframe(df: pd.DataFrame, search_term: str, searchable_columns: List[str]) -> pd.DataFrame:
    """
    Search through DataFrame columns for the search term
    
    Args:
        df: pandas DataFrame to search
        search_term: string to search for
        searchable_columns: list of column names to search in
        
    Returns:
        Filtered DataFrame
    """
    if not search_term:
        return df
    
    # Convert search term to lowercase for case-insensitive search
    search_term_lower = search_term.lower()
    
    # Create boolean mask for rows that match search term
    mask = pd.Series([False] * len(df))
    
    for col in searchable_columns:
        # Convert column to string and search (case-insensitive)
        col_mask = df[col].astype(str).str.lower().str.contains(search_term_lower, na=False, regex=False)
        mask = mask | col_mask
    
    # Return filtered DataFrame
    filtered_df = df.loc[mask].copy()
    return filtered_df

def display_search_results(df: pd.DataFrame, search_term: str):
    """
    Display search results with highlighting and statistics
    
    Args:
        df: DataFrame with search results
        search_term: original search term
    """
    if df.empty:
        st.warning(f"No registration found for '{search_term}'")
        st.info("Please check the registration number and try again.")
        return
    
    # Display search statistics
    if len(df) == 1:
        st.success(f"✅ Found registration: {search_term}")
    else:
        st.success(f"Found {len(df)} records for '{search_term}'")
    
    # Display results in an expanded, easy-to-read format
    st.subheader("📋 Registration Details")
    
    # Show each result as a card-like display for better readability
    for record_num, (idx, row) in enumerate(df.iterrows()):
        with st.expander(f"Record {record_num + 1}" if len(df) > 1 else "Registration Information", expanded=True):
            # Display all columns and their values in a clean format
            col1, col2 = st.columns(2)
            
            filtered_col_idx = 0
            for column, value in row.items():
                # Convert column to string for comparison
                column_str = str(column)
                
                # Skip CAMP column
                if column_str.upper() == 'CAMP':
                    continue
                
                # Change column name display
                display_column = column_str
                if '/C Number' in column_str:
                    display_column = column_str.replace('/C Number', 'J/C')
                
                target_col = col1 if filtered_col_idx % 2 == 0 else col2
                with target_col:
                    st.write(f"**{display_column}:** {value}")
                
                filtered_col_idx += 1
    
    # Show raw data table as well
    st.subheader("📊 Raw Data")
    display_df = df.copy()
    
    # Remove CAMP column if it exists
    if 'CAMP' in display_df.columns:
        display_df = display_df.drop('CAMP', axis=1)
    
    # Rename /C Number to J/C if it exists
    display_df.columns = [str(col).replace('/C Number', 'J/C') if '/C Number' in str(col) else str(col) for col in display_df.columns]
    
    # Convert all columns to string to prevent Arrow issues
    for col in display_df.columns:
        display_df[col] = display_df[col].astype(str)
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Option to download filtered results
    if len(df) > 0:
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📥 Download registration details as CSV",
            data=csv_data,
            file_name=f"registration_{search_term}.csv",
            mime="text/csv"
        )

def main():
    """Main application function"""
    
    # Compact header
    st.title("🔍 Registration Search")
    
    # File upload in main area (compact)
    uploaded_file = st.file_uploader(
        "Upload Excel file",
        type=['xlsx', 'xls'],
        help="Upload .xlsx or .xls files"
    )
    
    if uploaded_file is not None:
        # Store the uploaded file in session state for persistence
        st.session_state['uploaded_file'] = uploaded_file
        st.session_state['file_name'] = uploaded_file.name
    
    # Check if we have a stored file
    if 'uploaded_file' in st.session_state and uploaded_file is None:
        uploaded_file = st.session_state['uploaded_file']
        col1, col2 = st.columns([3, 1])
        with col1:
             # Store the uploaded file in session state for persistence
            st.session_state['uploaded_file'] = uploaded_file
            st.session_state['file_name'] = uploaded_file.name
        
        # Check if we have a stored file
        if 'uploaded_file' in st.session_state and uploaded_file is None:
            st.info(f"Using previously uploaded file: {st.session_state['file_name']}")
            uploaded_file = st.session_state['uploaded_file']
            st.success(f"File loaded: {st.session_state['file_name']}")
        with col2:
            if st.button("🗑️ Clear"):
                del st.session_state['uploaded_file']
                del st.session_state['file_name']
                st.rerun()
    
    # Main search interface
    if uploaded_file is not None:
        # Load the Excel file
        with st.spinner("Loading..."):
            df = load_excel_file(uploaded_file)
        
        if df is not None:
            # Get searchable columns
            searchable_columns = get_searchable_columns(df)
            
            # Search bar (main focus)
            search_term = st.text_input(
                "Search:",
                placeholder="Enter registration number or any search term...",
                help="Search across all columns"
            )
            
            # Real-time search
            if search_term:
                with st.spinner("Searching..."):
                    filtered_df = search_dataframe(df, search_term, searchable_columns)
                    display_search_results(filtered_df, search_term)
            else:
                st.info("Enter a search term above to find records")
    
    else:
        st.info("Please upload an Excel file to start searching")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.stop()
