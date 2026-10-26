import pyaudio
import asyncio
import websockets
import numpy as np

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 16000  # 1 second chunks

async def stream_microphone():
    uri = "ws://127.0.0.1:8000/audio-stream"
    
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    print("Recording and streaming live audio... Press Ctrl+C to stop.")
    
    try:
        async with websockets.connect(uri) as websocket:
            while True:
                # Read raw bytes from the microphone
                data = await asyncio.to_thread(stream.read, CHUNK, False)
                
                # Convert to numpy array to check audio levels
                audio_data = np.frombuffer(data, dtype=np.int16)
                
                # Find the loudest single sample in the recording
                peak = int(np.abs(audio_data).max())
                
                # Create a dynamic visual volume bar (scales to terminal)
                # 32767 is the maximum possible amplitude for 16-bit audio
                volume_bars = int((peak / 32767) * 50)
                print(f"Mic Level: [{'#' * volume_bars}{'-' * (50 - volume_bars)}] Peak: {peak:5d}", end='\r')
                
                # Send the data to the FastAPI server
                await websocket.send(data)
                
    except Exception as e:
        print(f"\nStreaming stopped: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    try:
        asyncio.run(stream_microphone())
    except KeyboardInterrupt:
        print("\nStopped by user.")