# I tried to do this a fancier way with pyparsing but failed miserably...
# currently only returns "BAY" information
def get_amcc_dump(self, ip='10.0.1.4',slot=2,show_result=True):
    import subprocess
    result=subprocess.check_output(['amcc_dump','--all','10.0.1.4/2'])
    result_string=result.decode('utf-8')

    tablebreak='================================================================================'
    ipslotbreak='--------------------------------------------------------------------------------'

    ## break into tables
    split_result_string=result_string.split(tablebreak)
    # drop white space
    split_result_string = list(filter(None,[s for s in split_result_string if not s.isspace()]))

    amcc_dump_dict = {}
    # loop over tables in returned data
    for ii in range(0,len(split_result_string),2):
        header = split_result_string[ii]
        table = split_result_string[ii+1]

        split_header=header.split('|')
        split_header = list(filter(None,[s.lstrip().rstrip() for s in split_header if not s.isspace()]))
        sh0=split_header[0]

        ipslotbreakcnt=[]
        ipslotbreakcntr=0
        split_table=table.split('\n')
        for s in split_table:
            if ipslotbreak in s:
                ipslotbreakcntr+=1
            ipslotbreakcnt.append(ipslotbreakcntr)

        # loop over ip/slot combinations in returned data
        for jj in range(0,max(ipslotbreakcnt),2):
            this_ipslot_idxs=[ll for ll, xx in enumerate(ipslotbreakcnt) if xx in [jj,jj+1]]
            split_table_subset=np.array(split_table)[this_ipslot_idxs[1:]]
            split_table_subset=list(filter(None,[s.lstrip().rstrip() for s in split_table_subset if not s.isspace()]))
            ipslot=split_table_subset[0]
            table2=split_table_subset[2:]

            if 'RTM' in sh0 or 'Bay Raw GPIO' in sh0:
                continue

            split_ipslot=ipslot.split('|')
            split_ipslot = list(filter(None,[s.lstrip().rstrip() for s in split_ipslot if not s.isspace()]))
            ip=split_ipslot[0].split('/')[0]
            slot=split_ipslot[0].split('/')[1]

            if ip not in amcc_dump_dict.keys():
                amcc_dump_dict[ip]={}
            if int(slot) not in amcc_dump_dict[ip].keys():
                amcc_dump_dict[ip][int(slot)]={}

            if sh0 not in amcc_dump_dict[ip][int(slot)].keys():
                amcc_dump_dict[ip][int(slot)][sh0]={}

            #if sh0 is 'BAY':
            if sh0=="BAY":
                split_table2=table2
                split_table2=list(filter(None,[s.lstrip().rstrip() for s in split_table2]))
                for split_table3 in split_table2:
                    split_table3=split_table3.split('|')
                    split_table3 = list(filter(None,[s for s in split_table3]))
                    st3k=split_table3[0].lstrip().rstrip()                
                    if st3k not in amcc_dump_dict[ip][int(slot)][sh0].keys():
                        amcc_dump_dict[ip][int(slot)][sh0][st3k]={}
                    #add data
                    for kk in range(1,len(split_header)-1):
                        shkk=split_header[kk]
                        st3kk=split_table3[kk].lstrip().rstrip()
                        if shkk not in amcc_dump_dict[ip][int(slot)][sh0][st3k].keys():
                            amcc_dump_dict[ip][int(slot)][sh0][st3k][shkk]=st3kk

    if show_result:
        import json
        print(json.dumps(amcc_dump_dict, indent = 4))
    
    return amcc_dump_dict
