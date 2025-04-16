import streamlit as st
import boto3

st.title("Test boto3 in Streamlit")

try:
    s3 = boto3.client("s3")
    st.success("Successfully imported boto3 and created S3 client")
except Exception as e:
    st.error(f"Error: {e}")