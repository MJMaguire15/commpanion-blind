# dshow_list_and_set.py
# Enumerate DirectShow capture devices and query which props are supported.
# Requires: pip install comtypes pywin32

import comtypes
from comtypes import GUID, POINTER, HRESULT, COMMETHOD
from ctypes import c_long, c_int
from ctypes.wintypes import BOOL, ULONG, VARIANT_BOOL

# GUIDs / CLSIDs
CLSID_SystemDeviceEnum             = GUID("{62BE5D10-60EB-11d0-BD3B-00A0C911CE86}")
CLSID_FilterGraph                  = GUID("{E436EBB3-524F-11CE-9F53-0020AF0BA770}")
CLSID_VideoInputDeviceCategory     = GUID("{860BB310-5D01-11D0-BD3B-00A0C911CE86}")

IID_ICreateDevEnum                 = GUID("{29840822-5B84-11D0-BD3B-00A0C911CE86}")
IID_IBaseFilter                    = GUID("{56A86895-0AD4-11CE-B03A-0020AF0BA770}")
IID_IGraphBuilder                  = GUID("{56A868A9-0AD4-11CE-B03A-0020AF0BA770}")
IID_IAMVideoProcAmp                = GUID("{C6E13360-30AC-11D0-A18C-00A0C9118956}")
IID_IAMCameraControl               = GUID("{C6E13370-30AC-11D0-A18C-00A0C9118956}")

VideoProcAmp_Flags_Auto   = 0x0001
VideoProcAmp_Flags_Manual = 0x0002
CameraControl_Flags_Auto  = 0x0001
CameraControl_Flags_Manual= 0x0002

# VideoProcAmpProperty
VPA_Brightness=0; VPA_Contrast=1; VPA_Hue=2; VPA_Saturation=3; VPA_Sharpness=4
VPA_Gamma=5; VPA_ColorEnable=6; VPA_WhiteBalance=7; VPA_Backlight=8; VPA_Gain=9

# CameraControlProperty
CC_Pan=0; CC_Tilt=1; CC_Roll=2; CC_Zoom=3; CC_Exposure=4; CC_Iris=5; CC_Focus=6

class IAMVideoProcAmp(comtypes.IUnknown):
    _iid_ = IID_IAMVideoProcAmp
    _methods_ = [
        COMMETHOD([], HRESULT, 'Get',
                  (c_int, 'Property'),
                  (POINTER(c_long), 'pValue'),
                  (POINTER(c_long), 'pFlags')),
        COMMETHOD([], HRESULT, 'Set',
                  (c_int, 'Property'),
                  (c_long, 'lValue'),
                  (c_long, 'Flags')),
        COMMETHOD([], HRESULT, 'GetRange',
                  (c_int, 'Property'),
                  (POINTER(c_long), 'pMin'),
                  (POINTER(c_long), 'pMax'),
                  (POINTER(c_long), 'pSteppingDelta'),
                  (POINTER(c_long), 'pDefault'),
                  (POINTER(c_long), 'pCapsFlags')),
    ]

class IAMCameraControl(comtypes.IUnknown):
    _iid_ = IID_IAMCameraControl
    _methods_ = [
        COMMETHOD([], HRESULT, 'Get',
                  (c_int, 'Property'),
                  (POINTER(c_long), 'pValue'),
                  (POINTER(c_long), 'pFlags')),
        COMMETHOD([], HRESULT, 'Set',
                  (c_int, 'Property'),
                  (c_long, 'lValue'),
                  (c_long, 'Flags')),
        COMMETHOD([], HRESULT, 'GetRange',
                  (c_int, 'Property'),
                  (POINTER(c_long), 'pMin'),
                  (POINTER(c_long), 'pMax'),
                  (POINTER(c_long), 'pSteppingDelta'),
                  (POINTER(c_long), 'pDefault'),
                  (POINTER(c_long), 'pCapsFlags')),
    ]

def create_graph():
    from comtypes.client import CreateObject
    graph = CreateObject(CLSID_FilterGraph, interface=IGraphBuilder)
    return graph

def enum_video_devices():
    from comtypes.client import CreateObject
    dev_enum = CreateObject(CLSID_SystemDeviceEnum, interface=comtypes.IUnknown)
    dev_enum = dev_enum.QueryInterface(IID_ICreateDevEnum)
    class ICreateDevEnum(comtypes.IUnknown):
        _iid_ = IID_ICreateDevEnum
        _methods_ = [
            COMMETHOD([], HRESULT, 'CreateClassEnumerator',
                      (POINTER(GUID), 'clsidDeviceClass'),
                      (POINTER(POINTER(comtypes.IUnknown)), 'ppEnumMoniker'),
                      (c_long, 'dwFlags')),
        ]
    dev_enum = comtypes.cast(dev_enum, POINTER(ICreateDevEnum))

    enum_moniker = POINTER(comtypes.IUnknown)()
    hr = dev_enum.CreateClassEnumerator(CLSID_VideoInputDeviceCategory, comtypes.byref(enum_moniker), 0)
    if hr != 0 or not enum_moniker:
        return []

    # We’ll use COM helper from comtypes to walk IEnumMoniker
    from comtypes.moniker import EnumMoniker
    devices = []
    for moniker in EnumMoniker(enum_moniker):
        props = moniker.BindToStorage(None, None, comtypes.gen.Propertystore.IPropertyBag._iid_)
        # Fallback: use DisplayName
        b = moniker.GetDisplayName(None, None)
        name = str(b)
        devices.append((name, moniker))
    return devices

def main():
    from comtypes.client import CreateObject
    print("Enumerating DirectShow video devices…")
    devs = enum_video_devices()
    if not devs:
        print("No devices found."); return

    for idx,(name, moniker) in enumerate(devs):
        print(f"\n[{idx}] {name}")
        # Bind to filter
        IBaseFilter = comtypes.gen.DirectShowLib.IBaseFilter if hasattr(comtypes.gen, 'DirectShowLib') else None
        flt = moniker.BindToObject(None, None, IID_IBaseFilter)
        # Query interfaces
        vpa = flt.QueryInterface(IID_IAMVideoProcAmp) if flt else None
        cam = flt.QueryInterface(IID_IAMCameraControl) if flt else None

        def test_vpa(prop, label):
            try:
                mn= c_long(); mx=c_long(); st=c_long(); d=c_long(); caps=c_long()
                hr = vpa.GetRange(prop, mn, mx, st, d, caps)
                if hr==0:
                    print(f"  VideoProcAmp {label:<14}: range=({mn.value},{mx.value}) step={st.value} def={d.value} caps=0x{caps.value:X}")
            except Exception:
                pass

        def test_cam(prop, label):
            try:
                mn= c_long(); mx=c_long(); st=c_long(); d=c_long(); caps=c_long()
                hr = cam.GetRange(prop, mn, mx, st, d, caps)
                if hr==0:
                    print(f"  CameraCtrl  {label:<14}: range=({mn.value},{mx.value}) step={st.value} def={d.value} caps=0x{caps.value:X}")
            except Exception:
                pass

        if vpa:
            test_vpa(VPA_Brightness, "Brightness")
            test_vpa(VPA_Contrast,   "Contrast")
            test_vpa(VPA_Saturation, "Saturation")
            test_vpa(VPA_Sharpness,  "Sharpness")
            test_vpa(VPA_Gamma,      "Gamma")
            test_vpa(VPA_WhiteBalance,"WhiteBalance")
            test_vpa(VPA_Gain,       "Gain")
        else:
            print("  IAMVideoProcAmp not supported.")

        if cam:
            test_cam(CC_Exposure,   "Exposure")
            test_cam(CC_Focus,      "Focus")
            test_cam(CC_Zoom,       "Zoom")
        else:
            print("  IAMCameraControl not supported.")

if __name__ == "__main__":
    main()
