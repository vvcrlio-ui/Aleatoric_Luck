"""Read process-visible Windows CPU topology and available physical memory."""
import ctypes
from ctypes import wintypes
import json
import struct


def inventory():
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    size=wintypes.DWORD(0)
    kernel.GetLogicalProcessorInformation(None,ctypes.byref(size))
    buffer=ctypes.create_string_buffer(size.value)
    if not kernel.GetLogicalProcessorInformation(buffer,ctypes.byref(size)):raise ctypes.WinError(ctypes.get_last_error())
    masks=[]
    for offset in range(0,size.value,32):
        mask,relation=struct.unpack_from('QI',buffer.raw,offset)
        if relation==0:masks.append(mask & -mask)
    class Memory(ctypes.Structure):
        _fields_=[('length',wintypes.DWORD),('load',wintypes.DWORD)]+[(name,ctypes.c_ulonglong)
            for name in ('total_physical','available_physical','total_page','available_page','total_virtual','available_virtual','extended')]
    memory=Memory();memory.length=ctypes.sizeof(memory)
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(memory)):raise ctypes.WinError(ctypes.get_last_error())
    return {'physical_core_masks':masks,'physical_cores':len(masks),
        'physical_memory_bytes':memory.total_physical,'available_physical_memory_bytes':memory.available_physical}


if __name__=='__main__':print(json.dumps(inventory()))
