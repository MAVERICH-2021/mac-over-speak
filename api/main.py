import os
import sys
from datetime import datetime


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(os.path.dirname(PROJECT_ROOT), "mac_over_speak_api.log")


class TimestampedTee:
    def __init__(self, *streams):
        self.streams = streams
        self._buffer = ""

    def write(self, data):
        if not data:
            return 0

        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            formatted = f"[{timestamp}] {line}\n"
            for stream in self.streams:
                stream.write(formatted)
                stream.flush()
        return len(data)

    def flush(self):
        if self._buffer:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            formatted = f"[{timestamp}] {self._buffer}"
            for stream in self.streams:
                stream.write(formatted)
                stream.flush()
            self._buffer = ""
        for stream in self.streams:
            stream.flush()


def configure_logging():
    log_file = open(LOG_PATH, "a", buffering=1, encoding="utf-8")
    sys.stdout = TimestampedTee(sys.__stdout__, log_file)
    sys.stderr = TimestampedTee(sys.__stderr__, log_file)
    print("")
    print("=" * 72)
    print(f"API log session started: {datetime.now().isoformat(timespec='seconds')}")
    print(f"Writing logs to: {LOG_PATH}")

# Ensure the parent directory is in the path so we can import 'api'
sys.path.append(PROJECT_ROOT)

if __name__ == "__main__":
    configure_logging()
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'api.settings')
    from django.core.management import execute_from_command_line
    
    # Pre-load model if desired, or let it lazy load on first request
    # from api.asr_engine import asr_engine
    # asr_engine.load_model()
    
    print("Starting ASR Service on http://127.0.0.1:8333/transcribe/")
    execute_from_command_line([sys.argv[0], "runserver", "0.0.0.0:8333", "--noreload"])
