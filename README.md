This library enabled SDR processing on Nvidia GPUs.

This is in the tinkering phase, but bug reports and contributions are welcome.

https://chromium.googlesource.com/chromium/src/+/main/docs/linux/build_instructions.md

```git clone https://chromium.googlesource.com/chromium/tools/depot_tools.git```

Then add the depot_tools directory to the PATH environment variable.

```
mkdir webrtc
cd webrtc
fetch --nohooks webrtc
gclient sync
cd src
git checkout main
gn gen out/Default --args='is_debug=false is_clang=false'
autoninja api -C out/Default
```
