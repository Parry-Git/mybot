import pyaudio

def list_devices():
    p = pyaudio.PyAudio()
    print("\nAvailable Audio Devices:")
    for i in range(p.get_device_count()):
        try:
            dev = p.get_device_info_by_index(i)
            print(f"Index {i}: {dev['name']}")
            print(f"  Max Input Channels: {int(dev['maxInputChannels'])}")
            print(f"  Max Output Channels: {int(dev['maxOutputChannels'])}")
            print(f"  Default Sample Rate: {int(dev['defaultSampleRate'])}")
            print("-" * 20)
        except Exception as e:
            print(f"Could not get info for device {i}: {e}")
    
    try:
        default_output = p.get_default_output_device_info()
        print(f"\nDefault Output Device: {default_output['name']} (Index {default_output['index']})")
    except Exception as e:
        print(f"Could not get default output device info: {e}")
    p.terminate()

if __name__ == "__main__":
    list_devices()
