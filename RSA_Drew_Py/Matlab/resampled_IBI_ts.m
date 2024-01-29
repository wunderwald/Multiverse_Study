function [time_series_out]=resampled_IBI_ts(ibi_in, sample_freq, is_sec)

%this resamples the ibi_series into an equally spaced time series, ibi's
%that fall between samples have a weighted representation (i.e. 
%time_before_sample x ibi_before_sample + %time_after_sample x ibi_after_sample
%the sampling frequency should be given in Hz, and ibi data in ms
%(i.e. 4Hz is one sample every 250ms)

samp_interval=1000/sample_freq;

if is_sec
    ibi_in=ibi_in *1000;
end
    

length_of_data=length(ibi_in);

time_ser(1)=ibi_in(1);

for i=2:length_of_data
    time_ser(i)=time_ser(i-1)+ibi_in(i);
end

num_pts = ceil(time_ser(length(time_ser))/samp_interval);
%pause  %uncomment this line to check the legth of output vector

time_series_out=zeros(num_pts-1,2);

marker=1;
prev_beat=time_ser(1);
next_beat=time_ser(1);

for i=1:num_pts-1
    
    time_series_out(i,1)=samp_interval*i;
    if time_series_out(i,1)<next_beat
        time_series_out(i,2)=ibi_in(marker);
    else
       rt_side=time_series_out(i,1)-next_beat;
       lt_side=samp_interval-rt_side;
       time_series_out(i,2)=(rt_side/samp_interval)*ibi_in(marker+1)+(lt_side/samp_interval)*ibi_in(marker);
       marker=marker+1;
       prev_beat=time_ser(marker-1);
       next_beat=time_ser(marker);
    end
end