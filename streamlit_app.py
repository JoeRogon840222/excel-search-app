def main():
    st.title("🔍 Registration Number Search")
    st.markdown("""
    Upload your Excel file and search for registration numbers to get detailed information.
    Simply type any registration number in the search box to find all related data.
    """)

    # Sidebar file uploader
    with st.sidebar:
        st.header("📁 File Upload")
        uploaded_file = st.file_uploader(
            "Choose an Excel file",
            type=['xlsx', 'xls'],
            help="Upload .xlsx or .xls files"
        )

        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")
            file_size = uploaded_file.size
            st.info(f"File size: {file_size:,} bytes")

            st.session_state['uploaded_file'] = uploaded_file
            st.session_state['file_name'] = uploaded_file.name

        if 'uploaded_file' in st.session_state and uploaded_file is None:
            st.info(f"Using previously uploaded file: {st.session_state['file_name']}")
            uploaded_file = st.session_state['uploaded_file']
            if st.button("🗑️ Clear File"):
                del st.session_state['uploaded_file']
                del st.session_state['file_name']
                st.rerun()

    if uploaded_file is not None:
        with st.spinner("Loading Excel file..."):
            df = load_excel_file(uploaded_file)

        if df is not None:
            st.subheader("📊 Dataset Information")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", len(df))
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                st.metric("File Type", uploaded_file.name.split('.')[-1].upper())

            searchable_columns = get_searchable_columns(df)

            col_info = pd.DataFrame({
                'Column Name': df.columns,
                'Data Type': df.dtypes.astype(str),
                'Non-Null Count': df.count(),
                'Searchable': [col in searchable_columns for col in df.columns]
            })

            st.subheader("📋 Column Information")
            st.dataframe(col_info.astype(str), use_container_width=True, hide_index=True)

            st.subheader("📋 Select Columns to Search")
            selected_columns = st.multiselect(
                "Choose which columns to search in:",
                options=searchable_columns,
                default=searchable_columns
            )

            if not selected_columns:
                st.warning("Please select at least one column to search in.")
                selected_columns = searchable_columns

            st.info(f"Searching in {len(selected_columns)} column(s): {', '.join(selected_columns)}")

            search_term = st.text_input(
                "Enter registration number:",
                placeholder="Type registration number to search...",
                help="Enter any registration number to find all related information"
            )

            if search_term:
                with st.spinner("Searching..."):
                    filtered_df = search_dataframe(df, search_term, selected_columns)
                    display_search_results(filtered_df, search_term)
            else:
                st.subheader("📄 Full Dataset")
                display_df = df.copy()

                if 'CAMP' in display_df.columns:
                    display_df = display_df.drop('CAMP', axis=1)

                display_df.columns = [
                    str(col).replace('/C Number', 'J/C') if '/C Number' in str(col) else str(col)
                    for col in display_df.columns
                ]

                for col in display_df.columns:
                    display_df[col] = display_df[col].astype(str)

                st.dataframe(display_df, use_container_width=True, hide_index=True)

                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="Download full dataset as CSV",
                    data=csv_data,
                    file_name=f"full_dataset_{uploaded_file.name.split('.')[0]}.csv",
                    mime="text/csv"
                )
    else:
        st.info("👆 Please upload an Excel file using the sidebar to get started.")

        st.subheader("📖 How to use this app:")
        st.markdown("""
        1. **Upload your Excel file** (.xlsx or .xls)
        2. **Enter a registration number** to search
        3. **View all details** for that registration
        4. **Download** results if needed
        """)

if __name__ == "__main__":
    main()

