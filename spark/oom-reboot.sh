#!/usr/bin/env bash
# Make the Spark reboot when it runs out of memory instead of hanging. Unlike the rest of
# spark/, this needs root:
#
#   sudo spark/oom-reboot.sh install     # or uninstall
#   spark/oom-reboot.sh status
#
# On 2026-10-15 a model server ran the Spark out of memory four times, and each time the
# machine hung, for minutes up to four and a half hours, because the OOM killer acted late
# or never. install sets up four ways to a reboot:
#   - the OOM killer panics instead of picking a process (vm.panic_on_oom), and a panic
#     reboots after 10 seconds (kernel.panic);
#   - a task stuck in the kernel for two minutes panics too (kernel.hung_task_panic); the
#     GPU driver's tasks were stuck like that in one of the hangs;
#   - clone-memwatch (spark/memwatch.sh) reboots after about a minute of the whole machine
#     waiting on memory, the hang the first two did not catch;
#   - systemd runs the board's hardware watchdog, which resets the machine if even PID 1
#     stops for a minute.
# Model servers do not start at boot, so the machine comes back clean.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SYSCTL=/etc/sysctl.d/90-clone-oom-reboot.conf
WATCH=/usr/local/sbin/clone-memwatch
UNIT=/etc/systemd/system/clone-memwatch.service
WDOG=/etc/systemd/system.conf.d/90-clone-watchdog.conf

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
need_root() { [ "$(id -u)" = 0 ] || die "run it with sudo"; }

case "${1:-status}" in
    install)
        need_root
        [ -r /proc/pressure/memory ] || die "this kernel has no /proc/pressure/memory, which clone-memwatch reads"
        cat > "$SYSCTL" <<'EOF'
# Reboot instead of hanging when memory runs out (clone, spark/oom-reboot.sh).
vm.panic_on_oom = 1
kernel.panic = 10
kernel.hung_task_panic = 1
EOF
        sysctl -q -p "$SYSCTL"
        # A copy owned by root: the service must not run a file the user account can change.
        install -m 0755 -o root -g root "$ROOT/spark/memwatch.sh" "$WATCH"
        cat > "$UNIT" <<EOF
[Unit]
Description=Reboot when the machine is stalled on memory (clone)

[Service]
ExecStart=$WATCH
Restart=always
RestartSec=5
OOMScoreAdjust=-1000
Nice=-10
MemoryMin=64M

[Install]
WantedBy=multi-user.target
EOF
        mkdir -p "$(dirname "$WDOG")"
        cat > "$WDOG" <<'EOF'
# The board's watchdog resets the machine if systemd stops answering for a minute (clone).
[Manager]
RuntimeWatchdogSec=60s
EOF
        systemctl daemon-reload
        systemctl daemon-reexec   # systemd reads RuntimeWatchdogSec only when it starts
        systemctl enable --now clone-memwatch.service
        "$0" status
        ;;
    uninstall)
        need_root
        systemctl disable --now clone-memwatch.service 2>/dev/null || true
        rm -f "$SYSCTL" "$WATCH" "$UNIT" "$WDOG"
        sysctl -q -w vm.panic_on_oom=0 kernel.panic=0 kernel.hung_task_panic=0
        sysctl -q --system
        systemctl daemon-reload
        systemctl daemon-reexec
        echo "removed"
        ;;
    status)
        for k in vm.panic_on_oom kernel.panic kernel.hung_task_panic kernel.hung_task_timeout_secs; do
            printf '%-30s %s\n' "$k" "$(sysctl -n "$k")"
        done
        printf '%-30s %s\n' clone-memwatch "$(systemctl is-active clone-memwatch.service 2>/dev/null || true)"
        printf '%-30s %s\n' "hardware watchdog" "$(systemctl show -p RuntimeWatchdogUSec --value)"
        printf '%-30s %s\n' "memory pressure now" "$(grep '^full' /proc/pressure/memory 2>/dev/null || echo unknown)"
        ;;
    *) die "usage: oom-reboot.sh install|uninstall|status" ;;
esac
