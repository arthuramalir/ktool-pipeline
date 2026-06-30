import os
import pandas as pd
import plotly.express as px

print("====== 🧵 Stitching Ktool Data Layers Together ======")

# 1. Load the 2D Layout Coordinates (Stage 4)
coords_path = "URDABAI/data/perception_coordinates.parquet"
if not os.path.exists(coords_path):
    raise FileNotFoundError(f"Missing {coords_path}. Run stage 4 first.")
df_coords = pd.read_parquet(coords_path)
print(f"Loaded {len(df_coords)} coordinates. Columns: {df_coords.columns.tolist()}")

# 2. Load the Text Content (Stage 3)
docs_path = "URDABAI/data/documents.parquet"
if os.path.exists(docs_path):
    df_docs = pd.read_parquet(docs_path)
    # Automatically look for common text column names
    text_col = next((c for c in ['text', 'text_excerpt', 'content', 'quote', 'description'] if c in df_docs.columns), df_docs.columns[1])
    # Keep only ID and text to avoid messy duplicate columns during merge
    df_docs = df_docs[['id', text_col]].rename(columns={text_col: 'human_text'})
    
    # Merge text into coordinates
    df = pd.merge(df_coords, df_docs, on='id', how='left')
    print(f"Merged text content successfully using column: '{text_col}'")
else:
    print(f"⚠️ Warning: {docs_path} not found. Map will not display narrative text strings.")
    df = df_coords
    df['human_text'] = "Text data missing"

# 3. Load the Clusters & Uncertainty (Stage 5)
# Trying default filename from your Stage 5 spec
cluster_path = "URDABAI/data/cluster_results.parquet" 
if os.path.exists(cluster_path):
    df_clusters = pd.read_parquet(cluster_path)
    
    # Identify which columns represent clusters and uncertainty
    cluster_col = next((c for c in ['cluster_id', 'cluster', 'assignment', 'consensus_cluster'] if c in df_clusters.columns), None)
    uncertainty_col = next((c for c in ['uncertainty_score', 'entropy', 'uncertainty'] if c in df_clusters.columns), None)
    
    # If standard columns aren't found, grab the first non-ID columns available
    if not cluster_col: 
        cluster_col = [c for c in df_clusters.columns if c != 'id'][0]
    
    # Prepare slice to merge
    cols_to_keep = ['id', cluster_col]
    if uncertainty_col: 
        cols_to_keep.append(uncertainty_col)
        df_clusters = df_clusters[cols_to_keep].rename(columns={cluster_col: 'cluster_assignment', uncertainty_col: 'uncertainty'})
    else:
        df_clusters = df_clusters[cols_to_keep].rename(columns={cluster_col: 'cluster_assignment'})
        df_clusters['uncertainty'] = 0.0 # fallback baseline
        
    df = pd.merge(df, df_clusters, on='id', how='left')
    print(f"Merged cluster labels successfully using column: '{cluster_col}'")
else:
    print(f"⚠️ Warning: {cluster_path} not found. Defaulting to single-color visualization.")
    df['cluster_assignment'] = "Unclustered"
    df['uncertainty'] = 0.0

# Fill any structural missing values safely for Plotly
df['cluster_assignment'] = df['cluster_assignment'].fillna('Unknown').astype(str)
df['human_text'] = df['human_text'].fillna('No text snippet found for this ID')

# 4. Generate the Human-Readable Interactive Plot
print("🎨 Rendering interactive map...")
fig = px.scatter(
    df, 
    x="x",                         # Matches your actual column 'x'
    y="y",                         # Matches your actual column 'y'
    color="cluster_assignment",    # Colors the islands cleanly
    hover_name="cluster_assignment",
    hover_data={
        "x": False,                # Hide coordinate geometry math from users
        "y": False,
        "id": True,                # Keep ID for validation traceability
        "uncertainty": ":.3f",     # Show normalized uncertainty score
        "human_text": True         # Print out the actual raw narrative string!
    },
    title=f"Ktool Human-Interpretation Map ({df['embedding_type'].iloc[0]} + {df['method'].iloc[0]})",
    template="plotly_white",
    width=1200,
    height=850
)

# Style tweaks to make text tooltips readable
fig.update_traces(
    marker=dict(size=6, opacity=0.7, line=dict(width=0.5, color='White'))
)
fig.update_layout(
    hoverlabel=dict(
        bgcolor="white",
        font_size=12
    ),
    legend_title_text="Identified Communities"
)

# 5. Save to disk
output_html = "figures/interactive_perception_map.html"
os.makedirs("figures", exist_ok=True)
fig.write_html(output_html)

print(f"🚀 Success! Map exported to: {output_html}")
print("Double-click the file to open it natively in your browser.")