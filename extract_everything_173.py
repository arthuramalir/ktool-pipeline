import json
import os
import requests

# 1. AUTHENTICATION & CONFIGURATION
AUTH_TOKEN = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MTE0LCJpYXQiOjE3ODE2OTIyNTQsImV4cCI6MTc4NDI4NDI1NH0.zn14uJI0uLP2WxPKyQokNLdnRCIpgYQBhKlioLMf_9I"
PLATFORM_ID = "173"
BASE_URL = "https://ktool.agirrecenter.eus/api"
HEADERS = {
    "Authorization": AUTH_TOKEN,
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json"
}

# 2. DEFINING EXPLICIT DEEP POPULATION ARRAYS FOR EVERY COMPONENT
# This tells Strapi exactly which relational tables to open up and extract.
DEEP_PROJECT_PARAMS = "&populate[0]=perceptions&populate[1]=thematic_areas&populate[2]=partners&populate[3]=lead_agent&populate[4]=agents"
DEEP_PILOT_PARAMS = "&populate[0]=perceptions&populate[1]=thematic_areas&populate[2]=partners&populate[3]=lead_agent&populate[4]=agents"
DEEP_PROTOTYPE_PARAMS = (
    "&populate[0]=perceptions"
    "&populate[1]=thematic_areas"
    "&populate[2]=partners"
    "&populate[3]=lead_agent"
    "&populate[4]=agents"
    "&populate[5]=interconnections"
    "&populate[interconnections][populate][0]=prototype" 
    "&populate[interconnections][populate][1]=pilot"
)
DEEP_AGENT_PARAMS = "&populate[0]=thematic_areas"
DEEP_INFO_PARAMS = "&populate[0]=thematic_areas"

MASTER_ENDPOINTS = {
    "agents": f"/agents?pagination[pageSize]=1000&pagination[page]=0&filters[parent_platform][id]={PLATFORM_ID}{DEEP_AGENT_PARAMS}",
    "projects": f"/projects?pagination[pageSize]=1000&pagination[page]=0&filters[parent_platform][id]={PLATFORM_ID}{DEEP_PROJECT_PARAMS}",
    "pilots": f"/pilots?pagination[pageSize]=1000&pagination[page]=0&filters[parent_platform][id]={PLATFORM_ID}{DEEP_PILOT_PARAMS}",
    "prototypes": f"/prototypes?pagination[pageSize]=1000&pagination[page]=0&filters[parent_platform][id]={PLATFORM_ID}{DEEP_PROTOTYPE_PARAMS}",
    "perceptions": f"/perceptions?pagination[pageSize]=1000&pagination[page]=0&filters[parent_platform][id]={PLATFORM_ID}&populate=*",
    "informations": f"/informations?pagination[pageSize]=1000&pagination[page]=0&filters[parent_platform][id]={PLATFORM_ID}{DEEP_INFO_PARAMS}",
    "challenges_opportunities": f"/challenges?pagination[pageSize]=1000&pagination[page]=0&filters[parent_platform][id]={PLATFORM_ID}&populate=*",
    "connections": f"/information-pattern-connections?pagination[pageSize]=1000&pagination[page]=0&filters[parent_platform][id]={PLATFORM_ID}&populate=*"
}

def extract_entire_ecosystem():
    output_dir = f"data/{PLATFORM_ID}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("========================================================================")
    # Target location: Ireland / Platform 173
    print(f"🚀 INITIATING FULL MASTER CRAWLER FOR PLATFORM {PLATFORM_ID}")
    print("========================================================================")
    
    for name, endpoint in MASTER_ENDPOINTS.items():
        url = f"{BASE_URL}{endpoint}"
        print(f"📥 Deep-harvesting collection: [{name.upper()}]...")
        
        try:
            response = requests.get(url, headers=HEADERS)
            
            # Handle standard Strapi naming variations for challenges/opportunities if endpoint differs
            if response.status_code == 404 and name == "challenges_opportunities":
                fallback_url = f"{BASE_URL}/challenge-opportunities?pagination[pageSize]=1000&filters[parent_platform][id]={PLATFORM_ID}&populate=*"
                response = requests.get(fallback_url, headers=HEADERS)

            if response.status_code == 200:
                payload = response.json()
                records = payload.get('data', [])
                records_count = len(records)
                
                # Check for empty platform connection tables to fallback cleanly
                if name == "connections" and records_count == 0:
                    fallback_conn_url = f"{BASE_URL}/information-pattern-connections?pagination[pageSize]=1000&populate=*"
                    fallback_res = requests.get(fallback_conn_url, headers=HEADERS)
                    if fallback_res.status_code == 200:
                        payload = fallback_res.json()
                        records_count = len(payload.get('data', []))

                output_path = f"{output_dir}/{name}.json"
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
                    
                print(f"     ✅ Successfully stored {records_count} complete records ➔ {output_path}")
            else:
                print(f"     ❌ Endpoint skipped ({response.status_code}): {response.text[:80]}")
                
        except Exception as e:
            print(f"     ❌ Connection dropped on element: {str(e)}")

    print("========================================================================")
    print("🎉 HARVEST COMPLETE: Your data/173/ directory is fully enriched.")
    print("========================================================================")

if __name__ == "__main__":
    extract_entire_ecosystem()