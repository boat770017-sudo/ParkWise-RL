"""CSV Logging utility for training and evaluation episode metrics."""

import os
import pandas as pd
from typing import Dict, List, Any

class CSVLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def log_episode(self, filename: str, episode_idx: int, metrics: Dict[str, Any], agent_name: str = "Agent"):
        """Append single episode summary metrics to CSV."""
        filepath = os.path.join(self.log_dir, filename)
        row = {
            'episode': episode_idx,
            'agent': agent_name,
            **metrics
        }
        df = pd.DataFrame([row])
        
        # Append to CSV if exists, else write header
        header = not os.path.exists(filepath)
        df.to_csv(filepath, mode='a', index=False, header=header)

    def log_benchmark_summary(self, filename: str, summary_rows: List[Dict[str, Any]]):
        """Write summary comparison dataframe to CSV."""
        filepath = os.path.join(self.log_dir, filename)
        df = pd.DataFrame(summary_rows)
        df.to_csv(filepath, index=False)
