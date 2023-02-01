#!/bin/bash
# 
# usage: source mpiopts.sh
#        mpirun $MPIOPTS -np 4 ./some_program
#
# Sets options to suppress certain Open MPI warnings and allow more
# processes to start than there are physical processors; useful for
# debugging.

if [[ -n "$DEBUG" ]]; then
    DEBUGOPTS="-x DEBUG"
fi

# Options appropriate for local desktop/laptop testing
export MPIOPTS="$DEBUGOPTS --mca opal_warn_on_missing_libcuda 0 --oversubscribe"

# export MPIOPTS="$DEBUGOPTS --mca opal_warn_on_missing_libcuda 0"
