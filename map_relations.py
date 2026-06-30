import json
import os

def unpack_flat_patterns_array():
    info_path = "data/173/informations.json"
    
    if not os.path.exists(info_path):
        print(f"❌ Error: Cannot find {info_path}")
        return

    with open(info_path, "r", encoding="utf-8") as f:
        payload = json.load(f).get("data", [])

    print("========================================================================")
    print("🎯 DECODING EMBEDDED PATTERNS ARRAY COMPONENT")
    print("========================================================================")
    print(f"📊 Total Quotes: {len(payload)}")

    found_any_pattern_elements = 0

    for item in payload:
        node_id = item.get("id")
        attrs = item.get("attributes", {})
        
        # Pull the flat list array we just discovered
        pats_list = attrs.get("patterns", [])
        
        if isinstance(pats_list, list) and len(pats_list) > 0:
            found_any_pattern_elements += 1
            
            # Print a detailed dive of the first few populated instances we find
            if found_any_pattern_elements <= 3:
                print(f"\n✅ [Quote ID {node_id}] has {len(pats_list)} item(s) in 'patterns' array:")
                print(f"   📄 Quote text: \"{attrs.get('quote', '')[:60]}...\"")
                
                for idx, element in enumerate(pats_list):
                    print(f"   ↳ Element [{idx}] Type: {type(element)}")
                    if isinstance(element, dict):
                        print(f"     • Keys: {list(element.keys())}")
                        print(f"     • Data Content: {json.dumps(element, ensure_ascii=False)[:120]}...")
                    else:
                        print(f"     • Value: '{element}'")

    print("\n========================================================================")
    print("📈 ARRAY MATRIX ANALYSIS")
    print("========================================================================")
    print(f" • Quotes containing populated pattern entries: {found_any_pattern_elements}/{len(payload)}")

if __name__ == "__main__":
    unpack_flat_patterns_array()