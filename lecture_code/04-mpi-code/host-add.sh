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

inithost="csel-cuda-01.cselabs.umn.edu"

if ! ssh -o PasswordAuthentication=no -o BatchMode=yes exit; then
    printf "Password-free login has not been set up\n"
    printf "Run the commands on the following page\n"
    printf "https://cse.umn.edu/cseit/self-help-guides/secure-shell-ssh"
    exit
fi

hosts="\
        csel-cuda-01.cselabs.umn.edu \
        csel-cuda-02.cselabs.umn.edu \
        csel-cuda-03.cselabs.umn.edu \
        csel-cuda-04.cselabs.umn.edu \
        csel-cuda-05.cselabs.umn.edu \
"
# 128.101.34.54
# 128.101.34.61
# 128.101.34.62
# 128.101.34.63
# 128.101.34.64

for h in $hosts; do
    ssh -oStrictHostKeyChecking=no $h "printf 'Done : %s\n' '$h'"
done
