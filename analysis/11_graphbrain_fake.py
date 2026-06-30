from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
from graph_utils import ANALYTICS_DIR, write_frame

def parse_hyperedge_tokens(expr_str: str) -> list[str]:
    """Deconstructs a Graphbrain-style string expression into token lists."""
    # Clean up outer parentheses
    clean = expr_str.strip()
    if clean.startswith("(") and clean.endswith(")"):
        clean = clean[1:-1]
    
    # Simple split by whitespace while keeping sub-clauses safe
    tokens = []
    current_token = []
    paren_depth = 0
    
    for char in clean:
        if char == "(":
            paren_depth += 1
            current_token.append(char)
        elif char == ")":
            paren_depth -= 0
            current_token.append(char)
        elif char == " " and paren_depth == 0:
            if current_token:
                tokens.append("".join(current_token))
                current_token = []
        else:
            current_token.append(char)
            
    if current_token:
        tokens.append("".join(current_token))
    return tokens

def match_graphbrain_pattern(expression: str, pattern: str) -> dict[str, str] | None:
    """
    Evaluates a hyperedge against a symbolic Graphbrain pattern rule.
    Binds uppercase tokens to variable names.
    """
    expr_tokens = parse_hyperedge_tokens(expression)
    pat_tokens = parse_hyperedge_tokens(pattern)
    
    # Strict structural comparison
    if len(expr_tokens) != len(pat_tokens):
        return None
        
    bindings = {}
    
    for e_tok, p_tok in zip(expr_tokens, pat_tokens):
        # Handle wildcards
        if p_tok == "*" or p_tok == "*/C" or p_tok == "*/P":
            continue
            
        # Handle strict operator matching (e.g. 'helped/P')
        if "/" in p_tok and not p_tok.split("/")[0].isupper():
            if e_tok.lower() != p_tok.lower():
                return None
            continue
            
        # Handle Variable Binding (e.g., ACTOR/C, SYSTEM/C)
        if "/" in p_tok and p_tok.split("/")[0].isupper():
            var_name = p_tok.split("/")[0]
            bindings[var_name] = e_tok.replace("_", " ")
            continue
            
        # Fallback exact match
        if e_tok.lower() != p_tok.lower():
            return None
            
    return bindings

def run_advanced_pattern_matching() -> None:
    print("🔮 Running Graphbrain Rule Inference Engine...")
    
    claims_path = Path("data/processed/173/test/analysis/socio_semantic_hypergraph_claims.csv")
    if not claims_path.exists():
        print("[ERROR] Run Day 10 first to generate your hypergraph expressions database!")
        return
        
    df_claims = pd.read_csv(claims_path)
    
    # 2. Define the Socsemics Advanced Rule Inference Bank
    # Capitalized terms are variables extracted dynamically.
# An expanded, robust rule bank to handle natural language variation
    INFERENCE_RULES = {
        # Matches: (helped/P catalyst beneficiary)
        "Direct_Impact_3_Token": "(helped/P CATALYST/C BENEFICIARY/C)",
        # Matches: (helped/P catalyst beneficiary context_phrase)
        "Direct_Impact_With_Context": "(helped/P CATALYST/C BENEFICIARY/C EXTRA/C)",
        
        # Matches identity links: ('re/P subject indicator)
        "Systemic_Identity_3_Token": "(’re/P SUBJECT/C SYSTEMIC_INDICATOR/C)",
        # Matches identity links with a modifier/context trailing
        "Systemic_Identity_With_Context": "(’re/P SUBJECT/C SYSTEMIC_INDICATOR/C EXTRA/C)",
        
        # Matches past-tense state transformations
        "Strategic_Transformation_3_Token": "(been/P INTERVENTION/C OUTCOME/C)",
        "Strategic_Transformation_With_Context": "(been/P INTERVENTION/C OUTCOME/C EXTRA/C)",
        
        # Action-oriented ecosystem drivers
        "Ecosystem_Catalyst_Action": "(allowed/P CATALYST/C ACTION/C EXTRA/C)"
    }
    
    matched_insights = []
    
    print("\nApplying symbolic patterns across 139 parsed nodes...")
    
    # 3. Match Evaluation Engine
    for _, row in df_claims.iterrows():
        expr = str(row["hypergraph_expression"])
        
        for rule_name, pattern_string in INFERENCE_RULES.items():
            variables = match_graphbrain_pattern(expr, pattern_string)
            
            if variables is not None:
                match_entry = {
                    "rule_triggered": rule_name,
                    "raw_sentence": row["raw_sentence"],
                    "hypergraph_expression": expr,
                    "anchored_project": row["anchored_target_label"],
                    **variables
                }
                matched_insights.append(match_entry)

    df_results = pd.DataFrame(matched_insights)
    
    if not df_results.empty:
        write_frame(df_results, "graphbrain_inferred_knowledge.csv")
        print(f"\n🚀 Success! Successfully isolated {len(df_results)} high-order structural dynamics.")
        
        print("\n=======================================================")
        print("   EXTRACTED STRUCTURAL INSIGHT REPORT (VAR BINDING)")
        print("=======================================================\n")
        
        for idx, match in df_results.head(5).iterrows():
            print(f"🔥 Rule Triggered: [{match['rule_triggered']}]")
            print(f"   Context: \"{match['raw_sentence']}\"")
            print(f"   Expression: {match['hypergraph_expression']}")
            
            # Print whatever dynamic variables were successfully bound
            for key, val in match.items():
                if key not in ["rule_triggered", "raw_sentence", "hypergraph_expression", "anchored_project"] and pd.notna(val):
                    print(f"     🔹 Variable [{key}]: {val}")
            print("-" * 60)
    else:
        print("[WARNING] Zero matches met the explicit symbolic patterns. Try loosening rule definitions.")

if __name__ == "__main__":
    run_advanced_pattern_matching()