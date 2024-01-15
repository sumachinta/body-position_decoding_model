addpath 'D:\MATLAB_Analysis\Pre+process\Fncs\Run_tuning\testing_model_helper_fns\whisker_position_decoding_model\github_files\raw_data'
load("params.mat");
load("run_speed.mat");
load("SpikeTimes.mat");
load("trial_times.mat");
load("whisker_position.mat");

% get bins
bin_res = .015;
bins = 0:bin_res:params.recording_endtime;
nBins = 11; %has to be an oddnumber % number of spike bins of size bin_res to use for decoding single feature bins

% bin feature and neuron data
whisker_binned = get_features_in_timebins(params,bins,whisker_position);
speed_binned = get_features_in_timebins(params,bins,run_speed);

% bin spike data
k = 1; spike = [];
for n = 1:length(params.spike.Neuron_ID)
    [spike(n,:)] = histcounts(SpikeTimes{n,1},bins);
    spike(n,:) = spike(n,:)/bin_res;
end

% Select data between startT & endT
BOOL =[];
for i = 1:length(bins)
    BOOL(i) = isbetween(bins(i),startT,endT);
end

% select neurons
all_units = 1:1:length(params.spike.Neuron_ID);
spike_select=[]; spike_select = spike(all_units,:);
[X,output_whisker,trial_start_idx,trial_end_idx] = get_design_matrix(spike_select,whisker_binned,BOOL,nBins-1);
[testdata_whisker,preddata_whisker,cvindices] = performdecoding(trial_start_idx,trial_end_idx,X,output_whisker);
whisker_decoding_all_units = get_errors(testdata_whisker,preddata_whisker);
disp(['Rsquare= ' num2str(whisker_decoding_all_units(1))]);
disp(['corr_coeff= ' num2str(whisker_decoding_all_units(2))]);
disp(['mean_err= ' num2str(whisker_decoding_all_units(2))]);


[X,output_speed,trial_start_idx,trial_end_idx] = get_design_matrix(spike_select,speed_binned,BOOL,nBins-1);
[testdata_speed,preddata_speed] = performdecoding(trial_start_idx,trial_end_idx,X,output_speed,cvindices);
speed_decoding_all_units = get_errors(testdata_speed,preddata_speed);
disp(['Rsquare= ' num2str(speed_decoding_all_units(1))]);
disp(['corr_coeff= ' num2str(speed_decoding_all_units(2))]);
disp(['mean_err= ' num2str(speed_decoding_all_units(2))]);

% save vars
% bin_res
% speed_decoding_all_units;
% whisker_decoding_all_units;
