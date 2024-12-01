import subprocess
import socket
import struct
import requests
import time


def get_public_ip(retry_interval=5, max_retries=3):
    try_count = 0
    while try_count < max_retries:
        try:
            response = requests.get('https://api.ipify.org')
            if response.status_code == 200:
                return response.text
        except requests.RequestException as e:
            print("Could not retrieve public IP, will try again in %ds" % retry_interval)
            time.sleep(retry_interval)
            try_count += 1
    return None

def get_network_info_mac(interface, info):
    word = 1 + len(info.split())
    command = "networksetup -getinfo %s | awk '/%s:/ {print $%d; exit}'" % (interface, info, word)

    # Run the command and capture the output
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()

    # Decode the output to a string and strip any whitespace
    return out.decode().strip()
    
interface = "Wi-Fi"
current_ip = get_public_ip()
gateway = get_network_info_mac(interface, "Router") 
subnet = get_network_info_mac(interface, "Subnet mask")

def increase_ip(ip_address):
    # Convert the IP address to a 32-bit packed binary format
    packed_ip = socket.inet_aton(ip_address)
    # Unpack it as an integer
    ip_number = struct.unpack("!L", packed_ip)[0]
    # Increase the IP number by one
    new_ip_number = ip_number + 1
    # Pack the new IP number and convert it back to the dotted-quad string format
    new_packed_ip = struct.pack("!L", new_ip_number)
    new_ip_address = socket.inet_ntoa(new_packed_ip)
    return new_ip_address

def is_available(ip_address):
    command = ["ping", "-c", "3", ip_address]
    
    return subprocess.call(command) != 0

def set_manual_ip_mac(interface, ip_address, gateway, subnet):
    # Turn off the interface
    subprocess.run(['networksetup', '-setmanual', interface, ip_address, subnet, gateway])

def get_new_ip():
    """
    Updates the current IP to a new one. The new IP will be the old one increased by one.
    This will only work when the IP of the interface is also used as public IP address.
    In order to avoid the usage of already assigned IP addresses it will change the IP and test
    the network connection in loop until it succeeds.
    Currently its only working on Mac OS.
    """

    # Calculate new IP
    global current_ip
    current_ip = increase_ip(current_ip)
    while not is_available(current_ip):
        current_ip = increase_ip(current_ip)
    
    # Update IP
    set_manual_ip_mac("Wi-Fi", current_ip, gateway, subnet)
    time.sleep(2)
    # Check if Internet is working
    new_ip = get_public_ip()
    if new_ip:
        print(f"IP updated to {new_ip}")
    else:
        get_new_ip()

# Example usage:
if __name__ == "__main__":
    print(is_available("141.13.32.128"))
