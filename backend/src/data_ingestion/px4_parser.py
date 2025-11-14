from .log_parser_base import LogParser, TelemetryRecord
from typing import List

class PX4Parser(LogParser):
    """Placeholder for PX4 ULog parser"""
    
    def parse(self) -> List[TelemetryRecord]:
        """Parse PX4 ULog file - Not implemented"""
        print("⚠️  PX4 parser not implemented yet")
        print("   Install: pip install pyulog")
        return []