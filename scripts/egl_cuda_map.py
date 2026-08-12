#!/usr/bin/env python3
"""Print the EGL device index for a CUDA ordinal (argv[1]).

EGL enumerates devices in PCI-bus order, which need not match CUDA's order;
MUJOCO_EGL_DEVICE_ID indexes the EGL list, so callers wanting "render on
CUDA GPU N" must translate. Uses EGL_CUDA_DEVICE_NV (0x323A) per device.
Run with __EGL_VENDOR_LIBRARY_FILENAMES pinned to the NVIDIA ICD so Mesa's
duplicate devices are excluded from the list being indexed.
"""
import ctypes
import sys

egl = ctypes.CDLL("libEGL.so.1")
egl.eglGetProcAddress.restype = ctypes.c_void_p


def proc(name, restype, *argtypes):
    p = egl.eglGetProcAddress(name.encode())
    if not p:
        raise RuntimeError(f"eglGetProcAddress failed for {name}")
    return ctypes.CFUNCTYPE(restype, *argtypes)(p)


EGLDeviceEXT = ctypes.c_void_p
EGLAttrib = ctypes.c_ssize_t
qdev = proc(
    "eglQueryDevicesEXT", ctypes.c_uint,
    ctypes.c_int, ctypes.POINTER(EGLDeviceEXT), ctypes.POINTER(ctypes.c_int),
)
qattr = proc(
    "eglQueryDeviceAttribEXT", ctypes.c_uint,
    EGLDeviceEXT, ctypes.c_int, ctypes.POINTER(EGLAttrib),
)

EGL_CUDA_DEVICE_NV = 0x323A
want = int(sys.argv[1])
devs = (EGLDeviceEXT * 32)()
n = ctypes.c_int(0)
if not qdev(32, devs, ctypes.byref(n)):
    sys.exit("eglQueryDevicesEXT failed")
for i in range(n.value):
    val = EGLAttrib(-1)
    if qattr(devs[i], EGL_CUDA_DEVICE_NV, ctypes.byref(val)) and val.value == want:
        print(i)
        sys.exit(0)
sys.exit(f"no EGL device with CUDA ordinal {want} (n={n.value})")
