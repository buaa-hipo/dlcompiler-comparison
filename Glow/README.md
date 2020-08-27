## Glow

### Install
- Replace the original glow/tools/loader/ExecutorCore.cpp with our modified file (ExecutorCore/ExecutorCore.cpp)
- re-compile Glow

### Run end-to-end evaluation
```bash
sh run_glow_end2end.sh
```
### Run per-layer tracting
```bash
# tracing, generate a json file
sh glow_tracing.sh

# parsing the json file
python glow_tracing_parser.py
```
