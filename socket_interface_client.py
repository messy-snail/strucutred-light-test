#!/usr/bin/env python3
# Copyright 2019, Pick-it NV.
# All rights reserved.
import argparse
import binascii
import socket
import json
import sys
import time
from collections import OrderedDict
from enum import Enum

MULT = 10000.0
RCV_MSG_SIZE = 64
INTERFACE_VERSION = 11

class Command(Enum):
  NO_COMMAND = -1
  CHECK_PICKIT_MODE = 0
  REQUEST_CALIBRATION = 10
  REQUEST_DETECTION = 20
  CAPTURE_IMAGE = 22
  PROCESS_IMAGE = 23
  REQUEST_NEXT_OBJECT = 30
  CHANGE_CONFIGURATION = 40
  TAKE_SNAPSHOT = 50
  BUILD_BACKGROUND_CLOUD = 60
  GET_PICK_FRAME_DATA = 70

class ResponseStatus(Enum):
  UNKNOWN_COMMAND = -99
  STATUS_RUNNING = 0
  STATUS_IDLING = 1
  STATUS_CALIBRATING = 2
  CALIBRATION_PLATE_FOUND = 10
  NO_CALIBRATION_PLATE_FOUND = 11
  OBJECTS_FOUND = 20
  NO_OBJECTS_FOUND = 21
  NO_IMAGE_CAPTURED = 22
  EMPTY_ROI = 23
  NO_VALID_OBJECTS = 24
  NO_PICKABLE_OBJECTS = 25
  IMAGE_CAPTURED = 26
  CONFIGURATION_SUCCEEDED = 40
  CONFIGURATION_FAILED = 41
  SAVE_SNAPSHOT_SUCCEEDED = 50
  SAVE_SNAPSHOT_FAILED = 51
  BUILD_BACKGROUND_CLOUD_SUCCEEDED = 60
  BUILD_BACKGROUND_CLOUD_FAILED = 61
  GET_PICK_FRAME_DATA_OK = 70
  GET_PICK_FRAME_DATA_FAILED = 71

class Request(object):
  def __init__(self, command=Command.NO_COMMAND, args=None):
    self.pos_x = args.pos_x if args is not None else 0
    self.pos_y = args.pos_y if args is not None else 0
    self.pos_z = args.pos_z if args is not None else 0
    self.rot_x = args.rot_x if args is not None else 0
    self.rot_y = args.rot_y if args is not None else 0
    self.rot_z = args.rot_z if args is not None else 0
    self.rot_w = args.rot_w if args is not None else 0
    self.command = command
    self.setup_id = args.setup_id if args is not None else 0
    self.product_id = args.product_id if args is not None else 0
    self.orientation_convention = args.orientation_convention if args is not None else 1  # Angle-axis (UR)
    self.interface_version = INTERFACE_VERSION

  def to_binary_string(self):
    return (
      int(self.pos_x * MULT).to_bytes(4, byteorder='big', signed=True) +
      int(self.pos_y * MULT).to_bytes(4, byteorder='big', signed=True) +
      int(self.pos_z * MULT).to_bytes(4, byteorder='big', signed=True) +
      int(self.rot_x * MULT).to_bytes(4, byteorder='big', signed=True) +
      int(self.rot_y * MULT).to_bytes(4, byteorder='big', signed=True) +
      int(self.rot_z * MULT).to_bytes(4, byteorder='big', signed=True) +
      int(self.rot_w * MULT).to_bytes(4, byteorder='big', signed=True) +
      int(self.command.value).to_bytes(4, byteorder='big', signed=True) +
      self.setup_id.to_bytes(4, byteorder='big', signed=True) +
      self.product_id.to_bytes(4, byteorder='big', signed=True) +
      self.orientation_convention.to_bytes(4, byteorder='big', signed=True) +
      self.interface_version.to_bytes(4, byteorder='big', signed=True)
    )

def receive_and_parse_data(conn):
  data = conn.recv(RCV_MSG_SIZE)

  return OrderedDict([
    ('pos_x', int.from_bytes(data[0:4], byteorder='big', signed=True) / MULT),
    ('pos_y', int.from_bytes(data[4:8], byteorder='big', signed=True) / MULT),
    ('pos_z', int.from_bytes(data[8:12], byteorder='big', signed=True) / MULT),
    ('rot_x', int.from_bytes(data[12:16], byteorder='big', signed=True) / MULT),
    ('rot_y', int.from_bytes(data[16:20], byteorder='big', signed=True) / MULT),
    ('rot_z', int.from_bytes(data[20:24], byteorder='big', signed=True) / MULT),
    ('rot_w', int.from_bytes(data[24:28], byteorder='big', signed=True) / MULT),
    ('object_age', int.from_bytes(data[28:32], byteorder='big', signed=True) / MULT),
    ('object_type', int.from_bytes(data[32:36], byteorder='big', signed=True)),
    ('object_dimensions', OrderedDict([
      ('x', int.from_bytes(data[36:40], byteorder='big', signed=True) / MULT),
      ('y', int.from_bytes(data[40:44], byteorder='big', signed=True) / MULT),
      ('z', int.from_bytes(data[44:48], byteorder='big', signed=True) / MULT),
    ])),
    ('objects_remaining', int.from_bytes(data[48:52], byteorder='big', signed=True)),
    ('status', ResponseStatus(int.from_bytes(data[52:56], byteorder='big', signed=True))),
    ('rotation_convention', int.from_bytes(data[56:60], byteorder='big', signed=True)),
    ('protocol_version', int.from_bytes(data[60:64], byteorder='big', signed=True))
  ])

def print_msg_data(data):
  for k, v in data.items():
    val = v if not isinstance(v, OrderedDict) else json.dumps(v)
    print('{0: <20}:\t{1}'.format(k, val))


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Pickit Socket interface client')
  parser.add_argument('pickit_ip', type=str)
  parser.add_argument('--mode', type=str, default="get-all-objects",
                      choices=["get-all-objects", "send-poses", "single-cmd"])
  parser.add_argument('--quiet', action='store_true',
                      help="Do not print response messages")
  parser.add_argument('--port', type=int, default=5001)
  parser.add_argument('--orientation-convention', type=int, default=1,
                      choices=range(1, 6))
  parser.add_argument('--product-id', type=int, default=1,
                      help="Product ID used when --mode=change-config")
  parser.add_argument('--setup-id', type=int, default=1,
                      help="Setup ID used when --mode=change-config")
  parser.add_argument('--pos-x', type=float, default=0,
                      help="X position that is sent to Pickit")
  parser.add_argument('--pos-y', type=float, default=0,
                      help="Y position that is sent to Pickit")
  parser.add_argument('--pos-z', type=float, default=0,
                      help="Z position that is sent to Pickit")
  parser.add_argument('--rot-x', type=float, default=0,
                      help="X orientation that is sent to Pickit")
  parser.add_argument('--rot-y', type=float, default=0,
                      help="Y orientation that is sent to Pickit")
  parser.add_argument('--rot-z', type=float, default=0,
                      help="Z orientation that is sent to Pickit")
  parser.add_argument('--rot-w', type=float, default=0,
                      help="W orientation that is sent to Pickit")
  parser.add_argument('--cmd', type=int, default=0,
                      help="Command. Only used when mode is 'single-cmd'.")
  args = parser.parse_args()

  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as conn:
    conn.connect((args.pickit_ip, args.port))

    if args.mode == "get-all-objects":
      req = Request(command=Command.CHECK_PICKIT_MODE)
      conn.sendall(req.to_binary_string())
      data = receive_and_parse_data(conn)
      if data['status'] != ResponseStatus.STATUS_RUNNING:
        sys.exit('Pickit is not in Robot mode, exiting.')

      req = Request(command=Command.REQUEST_DETECTION, args=args)
      conn.sendall(req.to_binary_string())
      data = receive_and_parse_data(conn)

      objects_remaining = data['objects_remaining']
      object_counter = 1

      if not args.quiet:
        print('--------- Object #{0} -----------'.format(object_counter))
        print_msg_data(data)

      while object_counter <= objects_remaining:
        req = Request(command=Command.REQUEST_NEXT_OBJECT, args=args)
        conn.sendall(req.to_binary_string())

        data = receive_and_parse_data(conn)
        object_counter = object_counter + 1
        if not args.quiet:
          print('--------- Object #{0} -----------'.format(object_counter))
          print_msg_data(data)

    elif args.mode == "send-poses":
      while True:
        req = Request(command=Command.NO_COMMAND, args=args)
        conn.sendall(req.to_binary_string())
        time.sleep(0.1)

    elif args.mode == "single-cmd":
      req = Request(command=Command(args.cmd), args=args)
      conn.sendall(req.to_binary_string())
      data = receive_and_parse_data(conn)

      if not args.quiet:
        print_msg_data(data)
