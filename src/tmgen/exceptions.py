class TMgenException(Exception):
    """Base exception for the TMgen library"""

    def __init__(self, *args):
        super().__init__(*args)
