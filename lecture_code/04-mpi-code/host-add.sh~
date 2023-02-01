#!/bin/bash
#
# Loops through Veggie cluster machines to add them to the known_hosts
# file in SSH. Without being in known hosts, MPI sessions that attempt
# to contact these machines will silently stall.
#
# To work properly, MPI and this script also need a public/private key
# pair set up using the instructions listed here:
#
# https://cse.umn.edu/cseit/self-help-guides/secure-shell-ssh
#
# under the "Key-based Authentication" section

hosts="\
        csel-broccoli.cselabs.umn.edu \
        csel-carrot.cselabs.umn.edu \
        csel-potato.cselabs.umn.edu \
        csel-radish.cselabs.umn.edu \
        csel-spinach.cselabs.umn.edu \
"
# hosts="128.101.34.54 128.101.34.62 128.101.34.64 128.101.34.61 128.101.34.63"

for h in $hosts; do
    ssh -oStrictHostKeyChecking=no $h "printf 'Done : %s\n' '$h'"
done
