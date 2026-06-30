import json
import os

def inspect_attribute_fill_rate():
    platform_id = "173"
    local_dir = f"data/{platform_id}"
    project_file = f"{local_dir}/projects.json"

    print("========================================================================")
    print(f"📊 INSPECTING DROP-DOWN DATA ENRICHMENT FOR PLATFORM {platform_id}")
    print("========================================================================")

    if not os.path.exists(project_file):
        print(f"❌ Cannot inspect attributes. Missing file: {project_file}")
        return

    with open(project_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    
    projects = payload.get("data", [])
    total_projects = len(projects)

    if total_projects == 0:
        print("⚠️ No project data found to inspect.")
        return

    # Track how many times these fields actually contain valid, populated data
    metrics = {
        "partners (Sectors)": 0,
        "thematic_areas": 0,
        "perceptions": 0,
        "lead_agent": 0,
        "impact_level": 0
    }

    for p in projects:
        attrs = p.get("attributes", {})
        
        # 1. Check Partners / Sectors
        partners = attrs.get("partners")
        if partners and (isinstance(partners, list) or (isinstance(partners, dict) and partners.get("data")) or isinstance(partners, str)):
            metrics["partners (Sectors)"] += 1
            
        # 2. Check Thematic Areas (nested Strapi relation)
        themes = attrs.get("thematic_areas", {})
        if isinstance(themes, dict) and themes.get("data"):
            metrics["thematic_areas"] += 1
        elif isinstance(themes, list) and len(themes) > 0:
            metrics["thematic_areas"] += 1

        # 3. Check Perceptions linked
        perceptions = attrs.get("perceptions", {})
        if isinstance(perceptions, dict) and perceptions.get("data"):
            metrics["perceptions"] += 1

        # 4. Check Lead Agent
        if attrs.get("lead_agent"):
            metrics["lead_agent"] += 1

        # 5. Check Impact Level
        if attrs.get("impact_level"):
            metrics["impact_level"] += 1

    # Print Report Card
    print(f"Total Project Records Analyzed: {total_projects}\n")
    print(f"{'Field Name':<25} | {'Populated Count':<15} | {'Fill Rate (%)':<15}")
    print("-" * 60)
    
    for field, count in metrics.items():
        fill_rate = (count / total_projects) * 100
        print(f"{field:<25} | {count:<15} | {fill_rate:.1f}%")
    print("-" * 60)
    
    print("\n💡 HOW TO INTERPRET THIS:")
    print(" • If Fill Rate is 0.0%, your script missed the data. We need to fix the API population nesting.")
    print(" • If Fill Rate is between 10% - 90%, the data is coming through, but some entries were left blank by users.")
    print(" • If Fill Rate is close to 100%, you have everything you need for deep network modeling.")

if __name__ == "__main__":
    inspect_attribute_fill_rate()