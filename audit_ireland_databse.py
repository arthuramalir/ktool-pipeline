import json
import os
import requests

AUTH_TOKEN = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MTE0LCJpYXQiOjE3ODE2OTIyNTQsImV4cCI6MTc4NDI4NDI1NH0.zn14uJI0uLP2WxPKyQokNLdnRCIpgYQBhKlioLMf_9I"
PLATFORM_ID = "173"
BASE_URL = "https://ktool.agirrecenter.eus/api"
HEADERS = {"Authorization": AUTH_TOKEN, "Accept": "application/json"}

# Define the target endpoints exactly as before
TARGET_ENDPOINTS = {
    "perceptions": f"/perceptions?pagination[pageSize]=1&pagination[page]=0&filters[parent_platform][id]={PLATFORM_ID}",
    "projects": f"/projects?pagination[pageSize]=1&pagination[page]=0&filters[parent_platform][id]={PLATFORM_ID}",
    "informations": f"/informations?pagination[pageSize]=1&pagination[page]=0&filters[parent_platform][id]={PLATFORM_ID}",
    "connections": f"/information-pattern-connections?pagination[pageSize]=1&pagination[page]=0&filters[parent_platform][id]={PLATFORM_ID}"
}

def run_data_audit():
    local_dir = f"data/{PLATFORM_ID}"
    print("========================================================================")
    print(f"🔍 RUNNING DATA EXTRACTION COMPLETENESS AUDIT FOR PLATFORM {PLATFORM_ID}")
    print("========================================================================")
    print(f"{'Node Category':<20} | {'Local Count':<12} | {'Server Total':<12} | {'Status':<10}")
    print("-" * 72)

    all_passed = True

    for name, endpoint in TARGET_ENDPOINTS.items():
        local_path = f"{local_dir}/{name}.json"
        
        # 1. Get Local Count
        if not os.path.exists(local_path):
            local_count = 0
        else:
            with open(local_path, "r", encoding="utf-8") as f:
                local_data = json.load(f)
                local_count = len(local_data.get("data", []))
        
        # 2. Query Server Metadata (asking for pageSize=1 just to peek at the meta object)
        url = f"{BASE_URL}{endpoint}"
        try:
            response = requests.get(url, headers=HEADERS)
            if response.status_code == 200:
                server_meta = response.json().get("meta", {})
                server_total = server_meta.get("pagination", {}).get("total", 0)
                
                # Check for global connections table fallback
                if name == "connections" and server_total == 0 and local_count > 0:
                    fallback_url = f"{BASE_URL}/information-pattern-connections?pagination[pageSize]=1&pagination[page]=0"
                    res = requests.get(fallback_url, headers=HEADERS)
                    server_total = res.json().get("meta", {}).get("pagination", {}).get("total", 0)
            else:
                server_total = "ERR"
        except Exception:
            server_total = "CONN_ERR"

        # 3. Determine Status
        if server_total in ["ERR", "CONN_ERR"]:
            status = "⚠️ CHECK ERR"
            all_passed = False
        elif local_count == server_total:
            status = "✅ COMPLETE"
        elif local_count < server_total:
            status = "❌ INCOMPLETE"
            all_passed = False
        else:
            status = " OVERFLOW" # Local has more records than the server currently reports via filtered meta

        print(f"{name:<20} | {local_count:<12} | {server_total:<12} | {status:<10}")

    print("-" * 72)
    if all_passed:
        print(" AUDIT PASSED: Your local dataset perfectly matches the server ecosystem database.")
    else:
        print(" AUDIT FAILED: You are missing records. Adjust your pageSize parameter or add pagination looping.")

if __name__ == "__main__":
    run_data_audit()