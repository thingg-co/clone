#!/usr/bin/env bash
# frank-memwatch: reboot the machine once it has been stalled on memory for about a minute.
# spark/oom-reboot.sh installs it as a root service; it is not meant to be run from here.
#
# /proc/pressure/memory's "full avg60" is the share of the last minute in which every
# runnable task was waiting for memory. Normal work, a model loading or a download filling
# the page cache, keeps it near 0. When the Spark ran out of memory it hung instead: the
# kernel kept swapping, the GPU driver's allocations failed, and the OOM killer acted hours
# later or never. LIMIT is the percent that counts as stalled.
#
# Nothing forks once the loop runs, since a fork can itself wait for memory then.
LIMIT=${LIMIT:-40}
[ -r /proc/pressure/memory ] || { echo "frank-memwatch: no /proc/pressure/memory in this kernel" >&2; exit 1; }
exec {nap}<> <(:)   # a pipe nobody writes to: read -t on it sleeps without a fork
echo "frank-memwatch: rebooting if memory full avg60 reaches ${LIMIT}%" > /dev/kmsg
while :; do
    read -r -t 10 -u "$nap" _ || true
    while read -r kind _ avg60 _; do
        [ "$kind" = full ] || continue
        pct=${avg60#avg60=}
        if [ "${pct%%.*}" -ge "$LIMIT" ]; then
            echo "frank-memwatch: memory full avg60 ${pct}%, rebooting" > /dev/kmsg
            echo s > /proc/sysrq-trigger   # write out what can be written
            read -r -t 5 -u "$nap" _ || true
            echo u > /proc/sysrq-trigger   # remount read-only
            read -r -t 2 -u "$nap" _ || true
            echo b > /proc/sysrq-trigger   # reboot now
        fi
    done < /proc/pressure/memory
done
