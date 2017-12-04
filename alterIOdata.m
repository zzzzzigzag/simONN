function [ input,target_output ] = alterIOdata( data_id,SNR,BER )

input = SNR(:,data_id);
target_output = BER(:,data_id);

end

