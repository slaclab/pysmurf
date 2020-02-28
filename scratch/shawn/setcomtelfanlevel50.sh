ssh root@shm-smrf-sp01 "clia setfanpolicy 20 3 disable"
ssh root@shm-smrf-sp01 "clia setfanpolicy 20 4 disable"
ssh root@shm-smrf-sp01 "clia setfanlevel all 50"
sleep 5
ssh root@shm-smrf-sp01 "clia fans"

