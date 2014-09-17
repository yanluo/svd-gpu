UTIL=sysmem_utilization,dram_utilization,tex_utilization,l2_utilization,\
shared_utilization,issue_slot_utilization,cf_fu_utilization,tex_fu_utilization,\
ldst_fu_utilization,single_precision_fu_utilization,special_fu_utilization\

BAND=shared_load_throughput,shared_store_throughput,gld_requested_throughput,\
gst_requested_throughput,gld_throughput,gst_throughput,tex_cache_throughput,\
l2_tex_read_throughput,l2_tex_write_throughput,l2_read_throughput,\
l2_write_throughput,dram_read_throughput,dram_write_throughput,\
sysmem_read_throughput,sysmem_write_throughput

EFFI=gld_efficiency,gst_efficiency,shared_efficiency,warp_execution_efficiency,\
warp_nonpred_execution_efficiency,sm_efficiency,branch_efficiency\

STALL=achieved_occupancy,stall_inst_fetch,stall_exec_dependency,stall_data_request,\
stall_texture,stall_sync,stall_other,stall_imc,stall_compute

if [ "$1" = 1 ]; then
    nvprof --metrics $UTIL ./svd -n 10000 -i 1
elif [ "$1" = 2 ]; then
    nvprof --metrics $BAND ./svd -n 10000 -i 1
elif [ "$1" = 3 ]; then
    nvprof --metrics $EFFI ./svd -n 10000 -i 1
elif [ "$1" = 4 ]; then
    nvprof --metrics $STALL ./svd -n 10000 -i 1
else
    nvprof ./svd -n 10000
fi
