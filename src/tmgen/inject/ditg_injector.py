#! /usr/bin/env/python


# coding=utf-8

import argparse
import json
from math import ceil
from subprocess import Popen

from tmgen.inject import InjectorBase


class DITGinjector(InjectorBase):
    """
    A wrapper around the D-ITG tool to inject traffic according 
    to a given traffic matrix.
    Traffic Matrix entries are interpreted as number of packets per second
    
    D-ITG: http://www.grid.unina.it/software/ITG/
    """

    def __init__(self, tm_fname, epoch_length, ID, destinations, scale_factor=1):
        super(DITGinjector, self).__init__(tm_fname, epoch_length, ID, destinations)
        self.recv_exec = 'ITGRecv'
        self.send_exec = 'ITGSend'
        self.scale_factor = scale_factor

    def _start_receiver(self):
        """ Starts a single receiver """
        self._receiver_process = Popen(self.recv_exec)
        return self._receiver_process

    def _start_senders(self):
        """
        Start multiple senders at each epoch
        :return: 
        """
        num_epochs = self.tm.num_epochs()
        # Each epoch new set of ITGSend will be started
        for e in range(num_epochs):
            # Get the TM values
            tm = self.tm.at_time(e)
            # Keep track of the subprocesses, we will need to wait on them
            processes = []
            # For each logical destination there will be multiple IP addresses
            for dstID, ips in self.destinations:
                # For each IP start a new sender, limited by the time of epoch_length
                for ip in ips:
                    p = Popen([self.recv_exec, '-t', self.epoch_length * 1000,
                               '-a', ip, '-T', 'UDP', '-d', 100, '-C',
                               ceil(tm[self.ID, dstID] * self.scale_factor)])
                    # Store process
                    processes.append(p)
            # Wait until epoch ends and all senders complete
            for p in processes:
                p.wait()

    def run(self):
        """ Execute the injection """
        r = self._start_receiver()
        self._start_senders()
        # Must kill the reciever
        r.terminate()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tm', help='The traffic matrix filename', required=True)
    parser.add_argument('-l', '--epoch-length', type=int, help='Length of an epoch, in seconds', required=True)
    parser.add_argument('-i', '--id', type=int, help='The integer ID of the traffic matrix node source entry')
    parser.add_argument('-s', '--scale', type=float, help='Scale the TM entries by this factor')
    parser.add_argument('-d', '--destinations', help='Mapping of integer IDs to IP addresses')
    options = parser.parse_args()

    DITGinjector(options.tm, options.epoch_length, options.id, json.loads(options.destinaiton),
                 scale_factor=options.scale)


if __name__ == '__main__':
    main()
