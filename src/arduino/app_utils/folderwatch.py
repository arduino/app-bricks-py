# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
import queue


# TODO: add support to event types other than file creation
class FolderWatcher:
    def __init__(self, path, patterns=["*"], ignore_patterns=[]):
        self._path = path
        self._observer = Observer()
        self._handler = FolderEventHandler(patterns=patterns, ignore_patterns=ignore_patterns, ignore_directories=True)

    def wait_for_event(self):
        return self._handler.wait_for_event()

    def start(self):
        self._observer.schedule(self._handler, self._path, recursive=True)
        self._observer.start()

    def produce(self):
        try:
            return self.wait_for_event()
        except Exception:
            return None

    def stop(self):
        self._observer.stop()
        self._observer.join()


class FolderEventHandler(PatternMatchingEventHandler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.queue = queue.Queue()

    def on_created(self, event):
        try:
            with open(event.src_path, "rb") as file:
                file_contents = file.read()
            self.queue.put(file_contents)
        except Exception as e:
            print(f"Error reading file {event.src_path}: {e}")
            raise

    def wait_for_event(self):
        return self.queue.get()

    def stop(self):
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except queue.Empty:
                break
