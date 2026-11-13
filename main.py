from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
from curator import PreferenceDatasetCurator

app = FastAPI(
    title="LLM Dataset Curation & Preference Alignment Pipeline",
    description="Automated validation and formatting pipeline for DPO/RLHF preference training pairs.",
    version="1.0.0"
)

curator = PreferenceDatasetCurator()

class PreferenceSample(BaseModel):
    prompt: str = Field(..., example="Explain quantum computing in simple terms.")
    chosen: str = Field(..., example="Quantum computing uses qubits that can exist in superposition...")
    rejected: str = Field(..., example="It's just super fast computers.")

class BatchCurationRequest(BaseModel):
    samples: List[PreferenceSample]

@app.post("/v1/dataset/curate", summary="Curate and Validate DPO Preference Dataset")
def curate_dataset(payload: BatchCurationRequest):
    try:
        raw_data = [
            {"prompt": s.prompt, "chosen": s.chosen, "rejected": s.rejected}
            for s in payload.samples
        ]
        
        df_curated = curator.curate_dataset_batch(raw_data)
        
        valid_count = int((df_curated["is_training_ready"] == True).sum())
        total_count = len(raw_data)
        
        return {
            "status": "success",
            "summary": {
                "total_samples": total_count,
                "training_ready_samples": valid_count,
                "curation_success_rate_pct": round((valid_count / total_count) * 100, 2) if total_count > 0 else 0.0
            },
            "curated_records": df_curated.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
