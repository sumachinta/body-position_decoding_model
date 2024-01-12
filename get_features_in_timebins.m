function feature_binned = get_features_in_timebins(params,bins,feature)
Fps = params.features.fps;
VidStartTime = params.features.VidStartTime;
VidEndTime = params.features.VidEndTime;
parfor b = 1:length(bins)-1
    PF1 = Time2Frame(bins(b),VidStartTime,VidEndTime,Fps);%
    PF2 = Time2Frame(bins(b+1),VidStartTime,VidEndTime,Fps);%
    if sum(~cellfun(@isempty,PF1)) && sum(~cellfun(@isempty,PF2))  % the time is recorded/not recorded by camera
        % get which trial
        trial = find(~cellfun(@isempty,PF1));
        % if isempty(feature{1,trial}) 
        %     feature_binned(b) = NaN;
        % else
            feature_binned(b) = mean(feature{1,trial}(PF1{1,trial}:PF2{1,trial},1));
        % end
    else
        feature_binned(b) = NaN;
    end       
end
end