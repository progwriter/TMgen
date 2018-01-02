#! /usr/bin/env/python

# coding=utf-8
from __future__ import print_function

import argparse
import json
import signal
import sys
from math import ceil
from subprocess import Popen

import time

from tmgen.inject import InjectorBase


def interrupt_handler(sig, frame):
    global injector
    if sig == signal.SIGINT:
        injector.stop()


signal.signal(signal.SIGINT, interrupt_handler)
SLEEP_GAP = 1  # should be enough to spin up receivers or kill senders


class DITGinjector(InjectorBase):
    """
    A wrapper around the D-ITG tool to inject traffic according
    to a given traffic matrix.
    Traffic Matrix entries are interpreted as number of packets per second
    D-ITG: http://www.grid.unina.it/software/ITG/
    """

    def __init__(self, tm_fname, epoch_length, ID, destinations, port, sig_port,
                 scale_factor=1):
        super(DITGinjector, self).__init__(tm_fname, epoch_length, ID,
                                           destinations)
        self.recv_exec = 'ITGRecv'
        self.send_exec = 'ITGSend'
        self.port = port
        self.sig_port = sig_port
        self.scale_factor = scale_factor

    def _start_receiver(self):
        """ Starts a single receiver """
        self._receiver_process = Popen(
            [self.recv_exec, '-Sp', str(self.sig_port)])
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
                           str(ceil(
                               tm[self.ID, int(dstID)] * self.scale_factor)),
                           '-rp', str(self.port),
                           '-Sdp', str(self.sig_port)],
                          stdout=None, stderr=sys.stderr)
                # Store process
                self._send_processes.append(p)
        # Wait until epoch ends and all senders complete
        while any(x.poll() is None for x in self._send_processes):
            for p in self._send_processes:
                p.wait()

    def run(self):
        """ Execute the injection """
        num_epochs = self.tm.num_epochs()
        for e in range(num_epochs):
            # Each epoch new set of ITGSend will be started
            r = self._start_receiver()
            # wait for receivers to spin up on all nodes
            time.sleep(SLEEP_GAP)
            self._start_senders(e)
            # wait for senders to shut down gracefully on all nodes
            time.sleep(SLEEP_GAP)
            # Must kill the reciever
            r.send_signal(signal.SIGINT)
            r.wait()

    def stop(self):
        print('Stopping the injector')
        for p in self._send_processes:
            p.terminate()
        self._receiver_process.terminate()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tm', help='The traffic matrix filename',
                        required=True)
    parser.add_argument('-l', '--epoch-length', type=float,
                        help='Length of an epoch, in seconds', required=True)
    parser.add_argument('-i', '--id', type=int,
                        help='The integer ID of the traffic matrix node source entry',
                        required=True)
    parser.add_argument('-s', '--scale', type=float,
                        help='Scale the TM entries by this factor',
                        default=1)
    parser.add_argument('-d', '--destinations',
                        help='Mapping of integer IDs to IP addresses')
    parser.add_argument('-p', '--port', type=int, help='DITG receiver port')
    parser.add_argument('--sigport', default=10435, type=int,
                        help='DITG singnalint port')
    options = parser.parse_args()

    injector = DITGinjector(options.tm, options.epoch_length, options.id,
                            json.loads(options.destinations),
                            options.port, options.sigport,
                            scale_factor=options.scale)
    injector.run()


if __name__ == '__main__':
    main()
