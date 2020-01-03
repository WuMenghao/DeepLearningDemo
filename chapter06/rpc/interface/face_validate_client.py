# -*- coding: utf-8 -*-
# Created by: WU MENGHAO
# Created on: 2019/12/31

from thrift.transport import TSocket, TTransport
from thrift.protocol.TBinaryProtocol import TBinaryProtocol
from chapter06.rpc.interface import face_validate_service
from chapter06.rpc.interface.ttypes import *
import argparse
import os


def call_remote_method(client, arg):
    method = arg.method
    user_id = arg.user
    face_dir = arg.face_dir

    if not os.path.exists(face_dir):
        raise Exception('face_dir not exist')

    if not method:
        raise Exception('method is not given')

    files = os.listdir(face_dir)

    if method == 'saveFaceEmb':
        return client.saveFaceEmb(faceImages=files, userId=user_id)

    if method == 'validateFace':
        return client.validateFace(faceImages=files, userId=user_id)


def init_client(host, port):
    socket = TSocket.TSocket(host, port)
    transport = TTransport.TFramedTransport(socket)
    protocol = TBinaryProtocol(transport)
    client = face_validate_service.Client(protocol)
    return client


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='localhost', help='the remote host to connect')
    parser.add_argument('--port', type=str, default='9090', help='the remote port to connect')
    parser.add_argument('--method', type=str, help='the method port to call')
    parser.add_argument('--user', type=int, help='id of user')
    parser.add_argument('--face_dir', type=str, help='face image dir')
    return parser.parse_args(argv)


def main(args):
    client = init_client(args.host, args.port)
    result = call_remote_method(client, args)
    print('result : %s' % result)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
