# coding=utf-8
from itertools import chain

import iperf3
from multiprocessing import Process


class IperfInjector(object):

    def __init__(self, ips_to_volumes, epoch_length, flow_size, listen_ports):
        """
        :param ips_to_volumes: a mapping of server IP addresses (as strings)
            to an array of volumes
        :param epoch_length: length of an epoch in seconds
        :param flow_size: average flow size 
        """
        # Ensure the number of epochs is consistent across all connections
        _s = set(map(len, ips_to_volumes.values()))
        assert len(_s) == 1
        self.num_epochs = _s.pop()
        self.epoch_length = epoch_length
        self.flow_size = flow_size
        self.volumes = ips_to_volumes
        self.listen_ports = listen_ports

    def _start_server(self, port):
        server = iperf3.Server(port)
        for _ in range(self.num_epochs):
            server.run()

    def _start_client(self, server_host, server_port):
        client = iperf3.Client()
        client.duration = self.epoch_length
        client.server_hostname = server_host
        client.server_port = server_port
        for e in range(self.num_epochs):
            client.num_streams = self.volumes[(server_host, server_port)][e]
            client.run()

    def run(self):
        servers = []
        for port in self.listen_ports:
            p = Process(self._start_server, args=(port,))
            p.start()
            servers.append(p)

        clients = []
        for i, (server_host, server_port) in self.volumes:
            p = Process(self._start_client, args=(server_host, server_port))
            p.start()
            clients.append(p)

        for p in chain(servers, clients):
            p.join()


if __name__ == '__main__':

    parser = argparse
