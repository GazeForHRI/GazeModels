import os
import socket
import msgpack
import msgpack_numpy as m
import numpy as np
import cv2 as cv
import argparse
from inference import estimate_from_clip

# ===== Settings =====
SHOW_CLIP_DEBUG = False     # set True to visualize each incoming clip in one window
CLIP_DEBUG_SCALE = 0.4      # scale factor when visualizing
CLIP_DEBUG_WAIT_MS = 1      # ms to show window (<=1 to go fast)

m.patch()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--socket", default="/tmp/mcgaze_server.sock",
                   help="UNIX socket path for this server")
    return p.parse_args()

def visualize_clip(frames, window_name="MCGaze Server Clip", scale=0.4, wait=1):
    """Debug-only: show frames side-by-side in a single window, then close."""
    if not frames:
        return
    resized = []
    for f in frames:
        if scale != 1.0:
            f = cv.resize(f, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
        resized.append(f)
    concat = np.concatenate(resized, axis=1)
    cv.imshow(window_name, concat)
    cv.waitKey(wait)
    cv.destroyWindow(window_name)


# --- MsgPack helpers ---
def send_msg(sock, obj):
    payload = msgpack.packb(obj, use_bin_type=True)
    sock.sendall(len(payload).to_bytes(4, "big"))
    sock.sendall(payload)

def recv_msg(sock):
    msg_len = int.from_bytes(sock.recv(4), "big")
    data = b""
    while len(data) < msg_len:
        packet = sock.recv(msg_len - len(data))
        if not packet:
            break
        data += packet
    return msgpack.unpackb(data, raw=False)


def _ensure_frames_list(crops_np):
    """
    Accepts np.array(head_crops) from client and returns List[np.ndarray(H,W,3), ...] in BGR uint8.
    The client already sends BGR frames; we just coerce types/shapes.
    """
    frames = []

    if isinstance(crops_np, list):
        itr = crops_np
    else:
        # crops_np may be an ndarray with shape (T,H,W,3) or object dtype
        itr = list(crops_np)

    for f in itr:
        if f is None:
            continue
        arr = np.asarray(f)
        # Ensure 3-channel uint8 (OpenCV style). If float, clip and convert.
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            frames.append(arr)
        else:
            # Skip malformed frames; you could also raise here
            continue
    return frames


def run_server():
    args = parse_args()
    SOCKET_FILE = args.socket
    # Fresh socket
    if os.path.exists(SOCKET_FILE):
        os.remove(SOCKET_FILE)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_FILE)
    server.listen(1)
    print(f"MCGaze server listening on {SOCKET_FILE}")

    while True:
        conn, _ = server.accept()
        try:
            msg = recv_msg(conn)
            clip_size = msg.get("clip_size", None)
            crops = msg.get("head_crops", [])

            # Convert to list of BGR frames
            frames = _ensure_frames_list(np.array(crops, dtype=object))
            T = len(frames)
            print(f"📥 Received clip with {T} frame(s); declared clip_size={clip_size}")

            # Debug view (optional)
            if SHOW_CLIP_DEBUG and T > 0:
                visualize_clip(frames, scale=CLIP_DEBUG_SCALE, wait=CLIP_DEBUG_WAIT_MS)

            results = []
            if T == 0:
                send_msg(conn, results)
                continue

            # Run MCGaze exactly once on the provided clip
            # We process exactly the frames the client sent (Design A); do not pad/duplicate.
            try:
                gaze_np = estimate_from_clip(frames)  # shape (T, 3), unit vectors
                gaze_np = gaze_np[:, [2, 0, 1]] # swap basis.

                for i in range(gaze_np.shape[0]):
                    results.append({
                        "gaze_detected": True,
                        "gaze_vector": gaze_np[i].tolist()
                    })
            except Exception as e:
                print("Inference error:", e)
                # Return a failure entry for each input frame to keep alignment
                results = [{"gaze_detected": False, "gaze_vector": None} for _ in range(T)]

            send_msg(conn, results)

        except Exception as e:
            print("Error while handling request:", e)
            # Try sending an empty result so the client doesn’t hang
            try:
                send_msg(conn, [])
            except Exception:
                pass
        finally:
            conn.close()


if __name__ == "__main__":
    run_server()
