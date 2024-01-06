import json
from typing import Dict

def load_col_types() -> Dict:
    col_types: Dict
    
    with open("col_dtype.json", "r") as f: 
        col_types = json.load(f)
        
    return col_types