#!/bin/bash

source shawnhammerfunctions

crate_id=3
slots_in_configure_order=(2 3 4)

## start parallel method
# setup stages
# 0 = carriers off.
# 1 = carrier eth responds to ping.
setup_complete=false
completion_status=3
declare -a slot_status=( $(for slot in ${slots_in_configure_order[@]}; do echo 2; done) )
while [[ "${setup_complete}" = false ]] ; do 
	for slot_idx in `seq 0 $((${#slots_in_configure_order[@]}-1))`; do 
	    slot=${slots_in_configure_order[$slot_idx]}

	    # Wait for gui to come up
	    if [ "${slot_status[${slot_idx}]}" = "3" ]; then
	    fi

	    ###########################################
	    echo "slot_status="${slot_status[@]}
	    # check if complete
	    status_summary=(`echo ${slot_status[@]} | tr ' ' '\n' | sort | uniq`)
	    echo ${#status_summary[@]}
	    echo ${status_summary[0]}
	    echo ${completion_status}
	    # break out of setup loop once all slot statuses reach completion status.
	    if [[ "${#status_summary[@]}" = "1" && "${status_summary[0]}" = "${completion_status}" ]] ; then
		setup_complete=true
	    fi
	done
done

