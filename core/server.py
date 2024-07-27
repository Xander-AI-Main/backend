import socket
import json

def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', 12345))
    server_socket.listen(1)

    print("Server is waiting for a connection...")
    client_socket, address = server_socket.accept()
    print(f"Connected to {address}")

    while True:
        data = client_socket.recv(1024).decode()
        if not data:
            break
        epoch_info = json.loads(data)
        print(epoch_info)

    client_socket.close()
    server_socket.close()

if __name__ == "__main__":
    start_server()

