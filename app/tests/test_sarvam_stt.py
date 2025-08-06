import asyncio
import wave
import logging

from sarvam_stt_stream import SarvamSTTStreamHandler

logging.basicConfig(level=logging.INFO)

# ====== CONFIG ======
AUDIO_FILE = "test_audio.mp3"  # Path to your audio file (must be 8kHz, mono, 16-bit PCM)
CHUNK_MS = 320  # e.g., 20ms at 16kHz mono = 640 bytes, at 8kHz mono = 320 bytes
# ====================

async def main():
    handler = SarvamSTTStreamHandler()
    await handler.start_stream()
    print("🔊 Streaming audio to Sarvam...")

    # Open audio file (8kHz, mono, 16-bit PCM WAV)
    with wave.open(AUDIO_FILE, "rb") as wf:
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        n_channels = wf.getnchannels()
        assert sample_rate == 8000, f"File must be 8kHz, got {sample_rate}"
        assert sample_width == 2, f"File must be 16-bit PCM, got {sample_width * 8}-bit"
        assert n_channels == 1, f"File must be mono, got {n_channels} channels"

        chunk_size = int(sample_rate * sample_width * CHUNK_MS / 1000)

        async def send_audio():
            while True:
                data = wf.readframes(chunk_size // sample_width)
                if not data:
                    break
                await handler.send_audio_chunk(data)
                await asyncio.sleep(CHUNK_MS / 1000.0)  # Simulate real-time
            print("✅ Done streaming audio.")

        send_audio_task = asyncio.create_task(send_audio())

        # Collect and print transcripts live
        try:
            while not send_audio_task.done():
                try:
                    transcript, is_final, language_code = await asyncio.wait_for(handler.get_transcript(), timeout=5)
                    print(f"📝 [{language_code}] {'(final)' if is_final else ''}: {transcript}")
                except asyncio.TimeoutError:
                    continue
        except KeyboardInterrupt:
            print("Interrupted by user.")

    await handler.close_stream()

if __name__ == "__main__":
    asyncio.run(main())
