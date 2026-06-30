import os
import pandas as pd
import plotly.express as px

print("====== 🧵 Stitching Real Ktool Data Files ======")

# 1. Load coordinates
coords_path = "URDABAI/data/perception_coordinates.parquet"
df_coords = pd.read_parquet(coords_path)
print(f"Loaded coordinates ({len(df_coords)} nodes). Columns: {df_coords.columns.tolist()}")

# 2. Load text narratives
docs_path = "URDABAI/data/documents.parquet"
if os.path.exists(docs_path):
    df_docs = pd.read_parquet(docs_path)
    # Identify text column dynamically
    text_col = next((c for c in ['text', 'text_excerpt', 'content', 'quote'] if c in df_docs.columns), df_docs.columns[1])
    df_docs = df_docs[['id', text_col]].rename(columns={text_col: 'narrative_text'})
    df = pd.merge(df_coords, df_docs, on='id', how='left')
    print("Merged narrative documents successfully.")
else:
    df = df_coords
    df['narrative_text'] = "No text found"

# 3. Load overlap / community data
overlap_path = "URDABAI/data/perception_graph_overlap.parquet"
if os.path.exists(overlap_path):
    df_overlap = pd.read_parquet(overlap_path)
    print(f"Loaded overlap data. Columns: {df_overlap.columns.tolist()}")
    
    # Identify community or label columns dynamically
    color_col = next((c for c in ['graph_community', 'community', 'cluster'] if c in df_overlap.columns), None)
    if not color_col:
        # Fall back to the first non-id column available
        color_col = [c for c in df_overlap.columns if c != 'id'][0]
        
    # Merge community metrics into main dataframe
    df_overlap_slice = df_overlap[['id', color_col]]
    df = pd.merge(df, df_overlap_slice, on='id', how='left')
    df = df.rename(columns={color_col: 'visualization_color'})
    print(f"Using column '{color_col}' for plot coloring.")
else:
    print("⚠️ perception_graph_overlap.parquet not found. Color defaulting to embedding type.")
    df['visualization_color'] = df['embedding_type']

# Clean string conversions for visualization tooltips
df['visualization_color'] = df['visualization_color'].fillna('Unknown').astype(str)
df['narrative_text'] = df['narrative_text'].fillna('Node contains no raw text payload')

# 4. Generate Interactive Visualization
print("🎨 Rendering map...")

# Separate the regular nodes from the collapsed perception stars for easier styling
df['is_perception'] = df['id'].astype(str).str.contains('perception', case=False)
df['marker_size'] = df['is_perception'].map({True: 12, False: 5})
df['marker_symbol'] = df['is_perception'].map({True: 'star', False: 'circle'})

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
    title="Ktool Interactive Perception Space & Graph Community Map",
    template="plotly_white",
    width=1300,
    height=850
)

# Apply explicit visual upgrades for human parsing
fig.update_traces(marker=dict(opacity=0.7, line=dict(width=0.4, color='White')))
fig.update_layout(
    hoverlabel=dict(bgcolor="white", font_size=12),
    legend_title_text="Graph Space Attribute"
)

# 5. Export Map
output_file = "figures/ktool_interactive_map.html"
os.makedirs("figures", exist_ok=True)
fig.write_html(output_file)

print(f"🚀 Done! Open this file in your browser: {output_file}")