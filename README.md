# LLM Dataset Curation & Preference Alignment Pipeline

A high-performance Python framework engineered for automated validation, cleaning, and formatting of preference training datasets for DPO/RLHF model alignment. Designed to curate high-quality chosen-versus-rejected training pairs at scale.

## Architecture & Core Features
- **Preference Pair Validation:** Automated heuristics checking for response length, non-trivial differences, and structural integrity.
- **DPO/RLHF Formatting:** Transforms raw prompt-response pairs into training-ready data structures with comprehensive sample metrics.
- **RESTful Curation API:** Built with FastAPI and Pydantic for high-throughput dataset validation runs.

## Quick Start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python main.py
