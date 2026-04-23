import socket

# Define the socket file
SOCKET_FILE = '/tmp/internal_socket.sock'

def send_gaze_vector(socket_path, message):
    # Create a Unix domain socket
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

    # Connect to the server
    client.connect(socket_path)

    # Send a message
    client.sendall(message.encode('utf-8'))

    client.close()