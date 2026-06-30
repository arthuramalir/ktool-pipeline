import json
import os
import requests

# 1. AUTHENTICATION HEADER configuration
AUTH_TOKEN = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MTE0LCJpYXQiOjE3ODE2OTIyNTQsImV4cCI6MTc4NDI4NDI1NH0.zn14uJI0uLP2WxPKyQokNLdnRCIpgYQBhKlioLMf_9I"

# Target configuration for Platform 173 (Ireland)
PLATFORM_ID = "173"
BASE_URL = "https://ktool.agirrecenter.eus/api"
HEADERS = {
    "Authorization": AUTH_TOKEN,
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json"
}

# Explicit endpoints pointing cleanly to Ireland (Platform 173) instead of 156
TARGET_ENDPOINTS = {
    "perceptions": f"/perceptions?pagination[pageSize]=1000&pagination[page]=0&populate=*&filters[parent_platform][id]={PLATFORM_ID}",
    "projects": f"/projects?pagination[pageSize]=1000&pagination[page]=0&populate=*&filters[parent_platform][id]={PLATFORM_ID}",
    "informations": f"/informations?pagination[pageSize]=1000&pagination[page]=0&populate=*&filters[parent_platform][id]={PLATFORM_ID}",
    
    # Structural join network layer
    "connections": f"/information-pattern-connections?pagination[pageSize]=1000&pagination[page]=0&populate=*&filters[parent_platform][id]={PLATFORM_ID}"
}

def extract_all_platform_173():
    # Create output space specifically for project/platform 173
    output_dir = f"data/{PLATFORM_ID}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🚀 Initiating K-Tool Data Extraction Layer for Ireland (Platform {PLATFORM_ID})...")
    print(f"📂 Storing records inside: {output_dir}/\n" + "-"*72)
    
    for name, endpoint in TARGET_ENDPOINTS.items():
        url = f"{BASE_URL}{endpoint}"
        print(f"Pulling collection: [{name}]...")
        
        try:
            response = requests.get(url, headers=HEADERS)
            if response.status_code == 200:
                payload = response.json()
                records_count = len(payload.get('data', []))
                
                # If connections filter by platform ID returns empty because it's a global table,
                # we fall back to a full pull to ensure no loss of relational metadata links.
                if name == "connections" and records_count == 0:
                    print(f"     ⚠️ Platform filter returned 0 connections. Falling back to global relational pull...")
                    fallback_url = f"{BASE_URL}/information-pattern-connections?pagination[pageSize]=1000&pagination[page]=0&populate=*"
                    fallback_res = requests.get(fallback_url, headers=HEADERS)
                    if fallback_res.status_code == 200:
                        payload = fallback_res.json()
                        records_count = len(payload.get('data', []))

                output_path = f"{output_dir}/{name}.json"
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
                    
                print(f"     ✅ Saved {records_count} elements to {output_path}")
            else:
                # If a structural endpoint throws an error (e.g., if connections table isn't enabled for 173)
                print(f"     ❌ Failed ({response.status_code}): {response.text[:100]}")
        except Exception as e:
            print(f"     ❌ Error connecting to endpoint: {str(e)}")

if __name__ == "__main__":
    extract_all_platform_173()