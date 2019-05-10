nvprof --metrics \
    dram_write_transactions,\
dram_read_throughput,\
dram_write_throughput,\
global_store_requests,\
gst_efficiency,\
gst_requested_throughput,\
gst_throughput,\
gst_transactions,\
gst_transactions_per_request,\
gld_efficiency,\
gld_requested_throughput,\
gld_throughput,\
gld_transactions,\
gld_transactions_per_request,\
l2_utilization,\
l2_write_throughput,\
l2_write_transactions,\
l2_read_throughput,\
l2_read_transactions,\
local_load_throughput,\
sm_efficiency,\
warp_execution_efficiency,\
achieved_occupancy \
    ./KokkosBatched_Test_BlockTridiagDirect.exe -N 16384 -L 128 -B 7 |& tee log 


##   ./KokkosBatched_Test_BlockTridiagDirect.exe -N 4096 -L 100 -B  7 |& tee log 
