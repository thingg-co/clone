#!/usr/bin/env bash
# clone-wifiwatch: connect the Wi-Fi again when NetworkManager has given up on it.
# spark/headless.sh installs it as a root service that a timer runs every minute; it is not
# meant to be run from here.
#
# When a Wi-Fi handshake fails, NetworkManager takes the saved password for wrong and asks
# the logged-in user for a new one. On the Spark nobody is there to answer. The login
# screen's password box takes the request and holds it, and the Wi-Fi stays off until
# someone logs in. That happened after a restart on 2026-10-16, when the first handshakes
# timed out, and the Spark was off the network for 16 hours. Without a login screen to ask,
# NetworkManager stops connecting it by itself instead, with the same result.
#
# So when the Wi-Fi is waiting for a password, has failed, or is disconnected, on two checks
# in a row, this connects it again. NetworkManager then uses the saved password, which is
# all that logging in did. A radio switched off (nmcli radio wifi off) is left alone.
set -u
MARK=${MARK:-/run/clone-wifiwatch}   # not empty when the last check found the Wi-Fi down

dev=$(nmcli -t -f DEVICE,TYPE device 2>/dev/null | awk -F: '$2 == "wifi" { print $1; exit }')
[ -n "$dev" ] || exit 0
state=$(nmcli -g GENERAL.STATE device show "$dev" 2>/dev/null)
case "${state%% *}" in
    30|60|120) ;;   # disconnected, need-auth (waiting for a password), failed
    *) if [ -s "$MARK" ]; then : > "$MARK"; fi; exit 0 ;;
esac
if [ ! -s "$MARK" ]; then
    echo "$state" > "$MARK"   # act only if the next check finds it down too
    exit 0
fi
: > "$MARK"
echo "clone-wifiwatch: $dev is $state on two checks in a row; connecting it again"
nmcli --wait 60 device connect "$dev"
