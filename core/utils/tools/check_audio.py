import pyaudio

p = pyaudio.PyAudio()

print(f"PyAudio Version: {pyaudio.__version__}")
try:
    print(f"Host API Count: {p.get_host_api_count()}")

    # Usually Host API 0 is ALSA
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    print(f"Host API 0 Device Count: {numdevices}")

    for i in range(0, numdevices):
        dev_info = p.get_device_info_by_host_api_device_index(0, i)
        print(f"Device {i}: {dev_info.get('name')} (In: {dev_info.get('maxInputChannels')}, Out: {dev_info.get('maxOutputChannels')})")

except Exception as e:
    print(f"Error querying devices: {e}")

p.terminate()
