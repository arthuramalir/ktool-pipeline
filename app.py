import pandas as pd
from pathlib import Path
import plotly.express as px
import streamlit as st

# Set Streamlit page layout to wide mode
st.set_page_config(layout="wide")
st.title("🧠 Ktool Interactive Perception Map")

# 1. Setup absolute paths relative to where this app.py lives
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"

coords_file = DATA_DIR / "perception_coordinates.parquet"
docs_file = DATA_DIR / "documents.parquet"
overlap_file = DATA_DIR / "perception_graph_overlap.parquet"

# Verify files exist
if not coords_file.exists():
    st.error(f"❌ Missing critical file at: {coords_file}")
    st.stop()

# 2. Load Data Layers
df_coords = pd.read_parquet(coords_file)

if docs_file.exists():
    df_docs = pd.read_parquet(docs_file)
    text_col = next((c for c in ['text', 'text_excerpt', 'content', 'quote'] if c in df_docs.columns), df_docs.columns[1])
    df_docs = df_docs[['id', text_col]].rename(columns={text_col: 'narrative_text'})
    df = pd.merge(df_coords, df_docs, on='id', how='left')
else:
    df = df_coords
    df['narrative_text'] = "No text found"

if overlap_file.exists():
    df_overlap = pd.read_parquet(overlap_file)
    color_col = next((c for c in ['graph_community', 'community', 'cluster'] if c in df_overlap.columns), df_overlap.columns[1])
    df_overlap_slice = df_overlap[['id', color_col]]
    df = pd.merge(df, df_overlap_slice, on='id', how='left')
    df = df.rename(columns={color_col: 'visualization_color'})
else:
    df['visualization_color'] = "Default"

# Clean text strings for display
df['visualization_color'] = df['visualization_color'].fillna('Unknown').astype(str)
df['narrative_text'] = df['narrative_text'].fillna('No text payload available')

# Separate standard narrative entries from collapsed perception stars
df['is_perception'] = df['id'].astype(str).str.contains('perception', case=False)
df['marker_symbol'] = df['is_perception'].map({True: 'star', False: 'circle'})

# 3. Create the Plotly Figure
fig = px.scatter(
    df, 
    x="x", 
    y="y", 
    color="visualization_color",
    symbol="marker_symbol",
    hover_name="id",
    hover_data={
        "x": False, 
        "y": False, 
        "visualization_color": True,
        "narrative_text": True,
        "marker_symbol": False
    },
    template="plotly_white",
    height=750
)
fig.update_traces(marker=dict(opacity=0.7, line=dict(width=0.4, color='White')))

# 4. EXPLICITLY TELL STREAMLIT TO DRAW IT (This fixes the blank page)
st.plotly_chart(fig, use_container_width=True)