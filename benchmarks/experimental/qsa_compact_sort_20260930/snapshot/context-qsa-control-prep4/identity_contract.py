"""Plain CPU identity checks over actual future system command outputs."""
import re

PROCESS_NVML_MIB = 32384
DEVICE_NVML_MIB = 32384
MODEL_CGROUP_BYTES = 120 * 1024**3
UUIDS = ('GPU-54a5dec0-85a9-837b-c54d-4a3752c1b620',
         'GPU-94e7fdea-085e-eea4-86a5-83bbd6ad0883',
         'GPU-7aaea8dd-2241-2730-1234-96a9cd033e11',
         'GPU-72ca49d9-70ae-f881-70b3-a16295209e47')


def gpu_identity(process_text, device_text, workers):
    expected = {(row['pid'], row['physical_uuid']) for row in workers}
    expected_uuids = {row['physical_uuid'] for row in workers}
    if len(expected) != 4 or len(expected_uuids) != 4:
        raise ValueError('four exact worker/UUID bindings required')
    processes = []
    for line in process_text.splitlines():
        fields = [value.strip() for value in line.split(',')]
        if (len(fields) != 3 or not re.fullmatch(r'[0-9]+', fields[0])
                or not re.fullmatch(r'[0-9]+ MiB', fields[2])):
            raise ValueError('NVML process field unknown')
        pid, uuid, memory = int(fields[0]), fields[1], int(fields[2].split()[0])
        if not 0 < memory <= PROCESS_NVML_MIB:
            raise ValueError('NVML process memory zero/negative/over cap')
        processes.append({'pid': pid, 'uuid': uuid, 'used_MiB': memory})
    if (len(processes) != 4 or
            {(row['pid'], row['uuid']) for row in processes} != expected):
        raise ValueError('actual GPU process PID→UUID bijection differs')
    devices = {}
    for line in device_text.splitlines():
        fields = [value.strip() for value in line.split(',')]
        if len(fields) != 2 or not re.fullmatch(r'[0-9]+', fields[1]):
            raise ValueError('NVML device field unknown')
        uuid, memory = fields[0], int(fields[1])
        if uuid in devices or not 0 < memory <= DEVICE_NVML_MIB:
            raise ValueError('NVML device memory unknown/duplicate/over cap')
        devices[uuid] = memory
    if devices.keys() != expected_uuids:
        raise ValueError('exact four physical GPU device samples required')
    return {'processes': processes, 'devices_used_MiB': devices,
            'process_limit_MiB': PROCESS_NVML_MIB,
            'device_limit_MiB': DEVICE_NVML_MIB}


def listener_identity(text, api_pid):
    lines = text.splitlines()
    if (len(lines) != 1 or '127.0.0.1:8201' not in lines[0]
            or re.findall(r'pid=([0-9]+),', lines[0]) != [str(api_pid)]):
        raise ValueError('one loopback 8201 listener must belong exactly to API PID')
    return {'actual_listener_line': lines[0], 'api_pid': api_pid}
