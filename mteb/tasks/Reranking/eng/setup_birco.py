# calculate_metadata.py
from mteb import MTEB
from . import (
    BIRCOArguAnaReranking,
    BIRCOClinicalTrialReranking,
    BIRCODorisMaeReranking,
    BIRCORelicReranking,
    BIRCOWhatsThatBookReranking,
)

def main():

    tasks = [
        BIRCODorisMaeReranking(),
        BIRCOArguAnaReranking(),
        BIRCOClinicalTrialReranking(),
        BIRCOWhatsThatBookReranking(),
        BIRCORelicReranking(),
    ]
    
    for task in tasks:
        print(f"Calculating metadata for {task.metadata.name}")
        task.calculate_metadata_metrics()
        
if __name__ == "__main__":
    main()