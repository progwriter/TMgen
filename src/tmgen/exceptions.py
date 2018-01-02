class TMgenException(Exception):
    """ Base exception for the TMgen library """

    def __init__(self, *args):
        super(TMgenException, self).__init__(*args)
