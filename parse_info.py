import json
import pandas as pd

def transform_strapi_payload(raw_json_string):
    """
    Parses raw Strapi api/informations JSON and flattens it 
    into a structured Pandas DataFrame matching our Micro/Meso specifications.
    """
    # Load JSON safely
    try:
        payload = json.loads(raw_json_string)
    except json.JSONDecodeError as e:
        # Gracefully catch data truncations
        print(f"⚠️ Warning: JSON decoding error (checking for truncation): {e}")
        # Repairing strategy if string was cut off mid-route
        if not raw_json_string.endswith("]}"):
            print("🔧 Attempting string termination repair for malformed payload segment...")
            # Simple repair logic for clipboard cutoffs
            if '"locale"' in raw_json_string.splitlines()[-1]:
                raw_json_string += '}}]}'
            payload = json.loads(raw_json_string)

    data_list = payload.get("data", [])
    
    flattened_records = []
    for item in data_list:
        node_id = str(item.get("id"))
        attrs = item.get("attributes", {})
        
        # Extract and sanitize the text string
        raw_quote = attrs.get("quote", "")
        clean_quote = raw_quote.replace('"', '').strip() if raw_quote else ""
        
        flattened_records.append({
            "information_id": f"info_{node_id}",
            "text": clean_quote,
            "date": attrs.get("date"),
            "locale": attrs.get("locale"),
            # Plugs seamlessly into our pipeline architecture as placeholders 
            # until cross-referenced with your perceptions.json metadata mapping
            "perception_id": None, 
            "perception_label": None
        })
        
    df_info = pd.DataFrame(flattened_records)
    return df_info

# Example Execution with your live payload segment
if __name__ == "__main__":
    # (Assuming your sample raw text is stored in raw_payload variable)
    # df_info = transform_strapi_payload(raw_payload)
    # print(df_info.head())
    pass