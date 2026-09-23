"""Live feed + WASD teleop for a SIYI A8 mini over Ethernet.

Setup
-----
The camera sits at ``192.168.144.25`` and does no DHCP, so give the PC's
Ethernet adapter a static address on the same subnet (e.g. ``192.168.144.100``
/ ``255.255.255.0``, no gateway) before running this.

Controls (focus the video window)
---------------------------------
W/S      pitch up/down         A/D   yaw left/right
Shift    hold for full speed   Q/E   zoom out/in
C        re-center gimbal      Esc   quit

Keys are read as *held* state (``GetAsyncKeyState`` on Windows) rather than
from ``cv.waitKey``: waitKey only sees key-repeat events, whose ~500 ms initial
delay makes the gimbal stutter on every press.
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import os
import sys
import threading
import time

# Must be set before cv2 opens the stream.  UDP + no buffering keeps glass-to-
# glass latency around 150-250 ms; the FFMPEG default buffers several frames.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;udp|fflags;nobuffer|flags;low_delay"
)

import cv2 as cv  # noqa: E402

from siyi_sdk import connect_udp  # noqa: E402

WINDOW = "SIYI A8 mini"
VK = {"W": 0x57, "A": 0x41, "S": 0x53, "D": 0x44, "Q": 0x51, "E": 0x45, "C": 0x43,
      "SHIFT": 0x10, "ESC": 0x1B}


class LatestFrame:
    """Grab frames on a thread and keep only the newest.

    Reading on the UI thread lets frames queue up whenever drawing or control
    stalls, and the lag never recovers.
    """

    def __init__(self, url: str) -> None:
        self.cap = cv.VideoCapture(url, cv.CAP_FFMPEG)
        self.frame = None
        self.running = self.cap.isOpened()
        self._lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                with self._lock:
                    self.frame = frame

    def read(self):
        with self._lock:
            return self.frame

    def close(self) -> None:
        self.running = False
        self.cap.release()


def _window_focused() -> bool:
    hwnd = ctypes.windll.user32.GetForegroundWindow()
    buf = ctypes.create_unicode_buffer(256)
    ctypes.windll.user32.GetWindowTextW(hwnd, buf, 256)
    return buf.value == WINDOW


def held(key: str) -> bool:
    return bool(ctypes.windll.user32.GetAsyncKeyState(VK[key]) & 0x8000)


async def main(args: argparse.Namespace) -> None:
    if sys.platform != "win32":
        sys.exit("held-key input uses the Win32 API; this script is Windows-only")

    url = args.url or f"rtsp://{args.ip}:8554/main.264"
    print(f"opening {url} ...")
    feed = LatestFrame(url)
    if not feed.running:
        sys.exit(f"could not open {url} -- is the PC on 192.168.144.x? try: ping {args.ip}")

    client = await connect_udp(args.ip)
    fw = await client.get_firmware_version()
    print(f"connected, firmware: {fw}")

    cv.namedWindow(WINDOW, cv.WINDOW_NORMAL)
    last_cmd = None
    last_send = 0.0
    zoom_dir = 0
    prev_c = False
    try:
        while True:
            frame = feed.read()
            if frame is not None:
                cv.imshow(WINDOW, frame)
            cv.waitKey(1)  # pumps the HighGUI event loop; keys come from held()
            if cv.getWindowProperty(WINDOW, cv.WND_PROP_VISIBLE) < 1:
                break

            focused = _window_focused()
            if focused and held("ESC"):
                break

            yaw = pitch = 0
            new_zoom = 0
            if focused:
                speed = 100 if held("SHIFT") else args.speed
                yaw = (held("D") - held("A")) * speed * args.yaw_sign
                pitch = (held("W") - held("S")) * speed
                new_zoom = held("E") - held("Q")
                c = held("C")
                if c and not prev_c:
                    asyncio.create_task(client.one_key_centering())
                prev_c = c

            # Speed commands latch until the next one, so send on change -- and
            # re-send periodically since a dropped UDP "stop" would leave the
            # gimbal spinning.
            now = time.monotonic()
            if (yaw, pitch) != last_cmd or now - last_send > 0.2:
                await client.rotate_nowait(yaw, pitch)
                last_cmd, last_send = (yaw, pitch), now

            if new_zoom != zoom_dir:
                zoom_dir = new_zoom
                asyncio.create_task(client.manual_zoom(zoom_dir))

            await asyncio.sleep(0.01)
    finally:
        await client.rotate_nowait(0, 0)
        if zoom_dir:
            await client.manual_zoom(0)
        await client.close()
        feed.close()
        cv.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ip", default="192.168.144.25")
    p.add_argument("--url", help="RTSP override; newer firmware serves rtsp://<ip>:8554/video1")
    p.add_argument("--speed", type=int, default=40, help="gimbal speed 1-100 (Shift = 100)")
    p.add_argument("--yaw-sign", type=int, choices=(-1, 1), default=1,
                   help="pass -1 if A/D turn the wrong way")
    asyncio.run(main(p.parse_args()))
