function [X,output_feature,trial_start_idx,trial_end_idx] = get_design_matrix(spike_select,feature,BOOL,B)
%% Get design matrix

idx = find(BOOL==1);
trial_start_idx = []; trial_start_idx(:,1) = [1 find(diff(idx)>1)+1];
trial_end_idx = []; trial_end_idx(:,1) = [find(diff(idx)>1) length(idx)];
NB = [];
    parfor i = 1:sum(BOOL)
        % modify this line to select past or future only neural data
        NB(i,:,:) = spike_select(:,idx(i)-(B/2):idx(i)+(B/2));
  %     NB(i,:,:) = spike_select(:,idx(i)-(B-1):idx(i)); % past neural data only
  %     NB(i,:,:) = spike_select(:,idx(i):idx(i)+(B-1)); % future neural data only
        output_feature(i,1) = feature(idx(i));
    end
X = []; X = reshape(NB,[size(NB,1) size(NB,2)*size(NB,3)]);
end
