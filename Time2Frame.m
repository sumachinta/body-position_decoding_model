function Frame=Time2Frame(PTime,VidStartTime,VidEndTime,Fps)
%% inputs
% PTime: time in the experiment which you want to convert to frames. It can
% be array of times. [Nx1]
% Videostarttimes: [Nx1]
% Videoendtimes: [Nx1]
% Fps: features fps resoltion
%% outputs
% {1, N} cell with N is the number of trials in the video. Each cell has
% frame numbers .....
    Frame=[];
    for j = 1:length(VidStartTime)
    temp=[];
    for i = 1:length(PTime)
        if PTime(i)> VidStartTime(j)+0.002 && PTime(i)< VidEndTime(j)
            temp=[temp; round((PTime(i)-VidStartTime(j))*Fps)];
        end
    end
    Frame{j}=temp;
    end
end