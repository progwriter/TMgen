#! /usr/bin/env/python

# coding=utf-8
from __future__ import print_function

import argparse
import json
import signal
import sys
from math import ceil
from subprocess import Popen

from tmgen.inject import InjectorBase


def interrupt_handler(sig, frame):
    global injector
    if sig == signal.SIGINT:
        injector.stop()


signal.signal(signal.SIGINT, interrupt_handler)


class DITGinjector(InjectorBase):
    """
    A wrapper around the D-ITG tool to inject traffic according 
    to a given traffic matrix.
    Traffic Matrix entries are interpreted as number of packets per second
    
    D-ITG: http://www.grid.unina.it/software/ITG/
    """

    def __init__(self, tm_fname, epoch_length, ID, destinations, port, scale_factor=1):
        super(DITGinjector, self).__init__(tm_fname, epoch_length, ID, destinations)
        self.recv_exec = 'ITGRecv'
        self.send_exec = 'ITGSend'
        self.port = port
        self.scale_factor = scale_factor

    def _start_receiver(self):
        """ Starts a single receiver """
        self._receiver_process = Popen(self.recv_exec)
        return self._receiver_process

    def _start_senders(self, e):
        """
        Start multiple senders at each epoch
        :return: 
        """
        # Get the TM values
        tm = self.tm.at_time(e)
        # Keep track of the subprocesses, we will need to wait on them
        self._send_processes = []
        # For each logical destination there will be multiple IP addresses
        for dstID, ips in self.destinations.items():
            # For each IP start a new sender, limited by the time of epoch_length
            for ip in ips:
                p = Popen([self.send_exec, '-t', str(self.epoch_length * 1000),
                           '-a', ip, '-T', 'UDP', '-d', str(100), '-C',
                           str(ceil(tm[self.ID, int(dstID)] * self.scale_factor)),
                           '-rp', self.port],
                          stdout=sys.stdout, stderr=sys.stderr)
                # Store process
                self._send_processes.append(p)
        # Wait until epoch ends and all senders complete
        for p in self._send_processes:
            p.wait()

    def run(self):
        """ Execute the injection """
        num_epochs = self.tm.num_epochs()
        # Each epoch new set of ITGSend will be started
        for e in range(num_epochs):
            print('Epoch %d' % e)
            r = self._start_receiver()
            self._start_senders(e)
            # Must kill the reciever
            r.send_signal(signal.SIGINT)
            r.wait()

    def stop(self):
        self._receiver_process.send_signal(signal.SIGINT)
        for p in self._send_processes:
            p.send_signal(signal.SIGINT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tm', help='The traffic matrix filename', required=True)
    parser.add_argument('-l', '--epoch-length', type=int, help='Length of an epoch, in seconds', required=True)
    parser.add_argument('-i', '--id', type=int, help='The integer ID of the traffic matrix node source entry',
                        required=True)
    parser.add_argument('-s', '--scale', type=float, help='Scale the TM entries by this factor',
                        default=1)
    parser.add_argument('-d', '--destinations', help='Mapping of integer IDs to IP addresses')
    parser.add_argument('-p', '--port', type=int, help='DITG receiver port')
    options = parser.parse_args()

    injector = DITGinjector(options.tm, options.epoch_length, options.id, json.loads(options.destinations),
                            options.port,
                            scale_factor=options.scale)
    injector.run()


if __name__ == '__main__':
    main()
