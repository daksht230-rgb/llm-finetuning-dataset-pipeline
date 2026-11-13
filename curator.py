import pandas as pd
from typing import List, Dict, Any

class PreferenceDatasetCurator:
    def __init__(self, target_format: str = "dpo_jsonl"):
        self.target_format = target_format

    def validate_preference_pair(self, prompt: str, chosen: str, rejected: str) -> Dict[str, Any]:
        """Validates and filters preference training pairs for DPO/RLHF pipelines."""
        issues = []
        
        if not prompt or len(prompt.strip()) < 5:
            issues.append("Prompt is too short or empty.")
        if not chosen or len(chosen.strip()) < 10:
            issues.append("Chosen response is insufficient.")
        if not rejected or len(rejected.strip()) < 10:
            issues.append("Rejected response is insufficient.")
        if chosen.strip() == rejected.strip():
            issues.append("Chosen and rejected responses cannot be identical.")

        is_valid = len(issues) == 0
        return {
            "is_valid": is_valid,
            "validation_issues": issues,
            "pair_metrics": {
                "prompt_tokens_approx": len(prompt.split()),
                "chosen_tokens_approx": len(chosen.split()),
                "rejected_tokens_approx": len(rejected.split())
            }
        }

    def curate_dataset_batch(self, raw_dataset: List[Dict[str, str]]) -> pd.DataFrame:
        curated_records = []
        for idx, row in enumerate(raw_dataset):
            prompt = row.get("prompt", "")
            chosen = row.get("chosen", "")
            rejected = row.get("rejected", "")
            
            validation = self.validate_preference_pair(prompt, chosen, rejected)
            
            curated_records.append({
                "sample_id": f"PREF-{idx+1:03d}",
                "prompt": prompt,
                "chosen_response": chosen,
                "rejected_response": rejected,
                "is_training_ready": validation["is_valid"],
                "error_logs": " | ".join(validation["validation_issues"]) if validation["validation_issues"] else "NONE"
            })
            
        return pd.DataFrame(curated_records)
