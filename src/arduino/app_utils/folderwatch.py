# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from watchdog.observers import Observer
from watchdog.events import FileSystemEvent, PatternMatchingEventHandler
import queue
from typing import Any


# TODO: add support to event types other than file creation
class FolderWatcher:
    def __init__(self, path: str, patterns: list[str] = ["*"], ignore_patterns: list[str] = []) -> None:
        self._path = path
        self._observer = Observer()
        self._handler = FolderEventHandler(patterns=patterns, ignore_patterns=ignore_patterns, ignore_directories=True)

    def wait_for_event(self) -> bytes:
        return self._handler.wait_for_event()

    def start(self) -> None:
        self._observer.schedule(self._handler, self._path, recursive=True)
        self._observer.start()

    def produce(self) -> bytes | None:
        try:
            return self.wait_for_event()
        except Exception:
            return None

    def stop(self) -> None:
        self._observer.stop()
        self._observer.join()


class FolderEventHandler(PatternMatchingEventHandler):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.queue: queue.Queue[bytes] = queue.Queue()

    def on_created(self, event: FileSystemEvent) -> None:
        try:
            with open(event.src_path, "rb") as file:
                file_contents = file.read()
            self.queue.put(file_contents)
        except Exception as e:
            print(f"Error reading file {event.src_path}: {e}")
            raise

    def wait_for_event(self) -> bytes:
        return self.queue.get()

    def stop(self) -> None:
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except queue.Empty:
                break
